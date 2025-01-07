#!../../venv/bin/python
from tokenizers import Tokenizer # type: ignore
from tokenizers.models import BPE # type: ignore
from tokenizers.pre_tokenizers import Split # type: ignore
from tokenizers.trainers import BpeTrainer # type: ignore
from . import config
from _io import StringIO # type: ignore

def train_tokenizer(f: StringIO, extra_tokens: list[str] = []) -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Split(r"\s", "merged_with_previous") # 保留空白字符同时防止汉字后面多出空格
    trainer = BpeTrainer(
        special_tokens=list(config.SPECIAL_TOKENS.keys()),
        vocab_size=32768 - len(extra_tokens)
    )
    tokenizer.train_from_iterator(f, trainer=trainer)
    tokenizer.add_tokens(extra_tokens)
    return tokenizer

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m minilm2.utils.train_tokenizer <path> [<extra_tokens_path>]")
        exit(1)
    path = sys.argv[1]
    print(f"Training encoder from {path}")
    extra_tokens = []
    if len(sys.argv) > 2:
        extra_tokens = list(open(sys.argv[2]).read())
    tokenizer = train_tokenizer(open(path), extra_tokens)
    tokenizer.save("tokenizer.json")
    print("Tokenizer saved to tokenizer.json.")

