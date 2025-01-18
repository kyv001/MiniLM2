import torch
from tqdm import tqdm
from torch.nn import functional as F
from .model import LLM
from . import config
from tokenizers import Tokenizer # type: ignore

def build_context(history: list[tuple[str, str]], tokenizer: Tokenizer,
                max_length: int) -> torch.Tensor:
    ids = []
    human_prefix_ids = tokenizer.encode(config.HUMAN_PREFIX).ids
    ai_prefix_ids = tokenizer.encode(config.AI_PREFIX).ids
    separator_ids = tokenizer.encode("\n" * 3).ids
    for i in range(len(history)):
        turn = history[i]
        ids += human_prefix_ids + tokenizer.encode(turn[0]).ids + separator_ids
        ids += ai_prefix_ids + tokenizer.encode(turn[1]).ids
        if i < len(history) - 1:
            ids += separator_ids
    ids = ids[-max_length:]
    return torch.LongTensor(ids).unsqueeze(0)

def append_history(history: list[tuple[str, str]], role: str, text: str) -> list[tuple[str, str]]:
    if role == "human":
        history.append((text, ""))
    else:
        history[-1] = (history[-1][0], text)
    return history

if __name__ == '__main__':
    import sys
    import os
    import json
    if len(sys.argv) < 2:
        print('Usage: python -m minilm2.llm.eval_pretrained <config_path>')
        exit(1)
    config_path = sys.argv[1]
    config_dir = os.path.dirname(config_path) # 配置文件路径
    train_config = json.load(open(config_path))

    # 加载tokenizer并获取词表大小
    print("Loading tokenizer...")
    tokenizer = Tokenizer.from_file(os.path.join(config_dir, train_config['tokenizer_path']))
    vocab_size = tokenizer.get_vocab_size()
    print(f"==> Vocab size: {vocab_size}")

    # 根据配置文件创建模型
    print("Loading model...")
    model = LLM(
        vocab_size=vocab_size,
        dim=train_config['model_dim'],
        max_length=train_config['max_length'],
        n_heads=train_config['num_heads'],
        n_blocks=train_config['num_layers'],
        dropout=train_config['dropout']
    )
    # 统计参数量
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"==> Number of parameters: {params / 1e6:.2f}M")
    # 加载已有的检查点
    if train_config['checkpoint_file']:
        checkpoint_path = os.path.join(config_dir, train_config['checkpoint_file'])
        print(f"==> Loading checkpoint from {checkpoint_path}, step={train_config['checkpoint_step']}")
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True))

    # 将模型移动到显存并编译以加速推理
    model.to(config.DEVICE)
    model.compile()
    model.eval()

    torch.set_float32_matmul_precision('high') # 调整精度以加速推理
    history: list[tuple[str, str]] = []
    while True:
        text = ""
        while True:
            try:
                if not text:
                    text += input("Use '!exit' to quit, Enter then press ^D to submit> ") + "\n"
                else:
                    text += input("> ") + "\n"
            except EOFError:
                break
        text = text.strip()
        if text == "!exit":
            break
        # 加入历史记录
        history = append_history(history, "human", text)
        # 构建输入
        input_ids = build_context(history, tokenizer, train_config['max_length'])
        input_ids = input_ids.to(config.DEVICE)
        # 推理
        response = ""
        n_blankline = 0
        with torch.no_grad():
            while True:
                try:
                    output = model(input_ids)
                    logits = F.softmax(output[0][-1] / train_config['temperature'], dim=-1)
                    # 采样输出，取概率最高的n个进行加权随机采样
                    probs, indices = logits.topk(round(vocab_size * train_config['top_p']))
                    token_id = indices[torch.multinomial(probs, 1)]
                    input_ids = torch.cat([input_ids, token_id.unsqueeze(0)], dim=1)[:, -train_config['max_length']:] # 自回归生成
                    token = tokenizer.id_to_token(token_id.item())
                    if token == "\n":
                        n_blankline += 1
                        if n_blankline >= 3:
                            break
                    else:
                        n_blankline = 0
                    print(token, end="", flush=True)
                    response += token
                except KeyboardInterrupt:
                    print()
                    break
        # 加入历史记录
        history = append_history(history, "ai", response)
