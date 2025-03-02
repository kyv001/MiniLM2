from . import config
from transformers import AutoTokenizer, PreTrainedTokenizer  # type: ignore
import numpy as np
from tqdm import tqdm

def preprocess_openwebtext(text_path: str, bin_path: str, tokenizer: PreTrainedTokenizer):
    with open(text_path, 'r', encoding='utf-8') as f, open(bin_path, 'wb') as f_bin:
        n_blank_lines = 0
        text = ""
        for line in tqdm(f):
            line = line.strip()
            if line:
                text += line + " "
                continue
            if not text:
                continue
            n_blank_lines += 1
            if n_blank_lines >= 3:
                ids = tokenizer.encode(text) + [config.SPECIAL_TOKENS["<eos>"]]
                np.array(ids, dtype=np.uint16).tofile(f_bin)
                n_blank_lines = 0
                text = ""
        ids = tokenizer.encode(text) + [config.SPECIAL_TOKENS["<eos>"]]
        np.array(ids, dtype=np.uint16).tofile(f_bin)

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 4:
        print("Usage: python -m minilm2.utils.preprocess_openwebtext <encoder_path> <text_path> <bin_path>")
        exit(1)
    encoder_path = sys.argv[1]
    text_path = sys.argv[2]
    bin_path = sys.argv[3]
    tokenizer = AutoTokenizer.from_pretrained(encoder_path, trust_remote_code=True)
    preprocess_openwebtext(text_path, bin_path, tokenizer)
