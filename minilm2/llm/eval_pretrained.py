import torch
from tqdm import tqdm
from torch.nn import functional as F
from .model import LLM
from . import config

if __name__ == '__main__':
    import sys
    import os
    import json
    
    from tokenizers import Tokenizer # type: ignore
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
    while True:
        text = ""
        try:
            while True:
                if not text:
                    text = input("Use '\\' to enter multi-line input. Press Ctrl-D to quit. > ")
                else:
                    text += input("> ")
                if text[-1] != '\\':
                    break
                text = text[:-1] + "\n"
        except EOFError:
            break
        # 编码输入文本
        input_ids = tokenizer.encode(text).ids[-train_config['max_length']:]
        input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(config.DEVICE)
        # 推理
        with torch.no_grad():
            while True:
                try:
                    output = model(input_ids)
                    logits = F.softmax(output[0][-1] / train_config['temperature'], dim=-1)
                    # 采样输出，取概率最高的n个进行加权随机采样
                    probs, indices = logits.topk(round(vocab_size * train_config['top_p']))
                    token = indices[torch.multinomial(probs, 1)]
                    if token.item() == config.SPECIAL_TOKENS["<eos>"]: # 遇到结束符则停止
                        print()
                        break
                    input_ids = torch.cat([input_ids, token.unsqueeze(0)], dim=1)[:, -train_config['max_length']:] # 自回归生成
                    print(tokenizer.id_to_token(token.item()), end="", flush=True)
                except KeyboardInterrupt:
                    print()
                    break

