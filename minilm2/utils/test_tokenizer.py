from tokenizers import Tokenizer # type: ignore

def test_tokenizer(tokenizer: Tokenizer):
    while True:
        text = input("--> ")
        tokens = tokenizer.encode(text)
        ids = tokens.ids
        print(ids)
        color = 0
        for i in ids:
            color += 1
            print(f"\033[0;3{color}m{tokenizer.id_to_token(i)}\033[0m", end="")
            if color == 5:
                color = 0
        print()

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m minilm2.utils.test_tokenizer <path_to_tokenizer>")
        exit(1)
    tokenizer = Tokenizer.from_file(sys.argv[1])
    test_tokenizer(tokenizer)
