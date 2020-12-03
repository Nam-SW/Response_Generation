import argparse

from tokenizer import train_tokenizer

from utils import JsonManager


if __name__ == "__main__":
    json_manager = JsonManager(__file__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit_alphabet", default=6000)
    parser.add_argument("--vocab_size", default=8000)
    parser.add_argument("--remove_names")
    parser.add_argument("--data_dir")
    parser.add_argument("--save_dir", default="config")

    args = parser.parse_args()

    limit_alphabet = int(args.limit_alphabet)
    vocab_size = int(args.vocab_size)
    remove_names = json_manager.load(args.remove_names)
    data_dir = args.data_dir
    save_dir = args.save_dir

    print(f"train tokenizer - vocab_size: {vocab_size}")
    train_tokenizer(
        limit_alphabet=limit_alphabet,
        vocab_size=vocab_size,
        remove_names=remove_names,
        data_dir=data_dir,
        save_dir=save_dir,
    )
