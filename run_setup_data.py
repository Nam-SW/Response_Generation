import argparse

from dataloader.preprocessing import setup_data
from utils import JsonManager


if __name__ == "__main__":
    json_manager = JsonManager(__file__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_dir")
    parser.add_argument("--preprocessed_data_dir")
    parser.add_argument("--model_use_data_dir")
    parser.add_argument("--remove_names")
    parser.add_argument("--tokenizer_config")
    parser.add_argument("--utterance_size", default=4)
    parser.add_argument("--max_len", default=100)
    parser.add_argument("--use_multiprocessing", default=True)

    args = parser.parse_args()

    raw_data_dir = args.raw_data_dir
    preprocessed_data_dir = args.preprocessed_data_dir
    model_use_data_dir = args.model_use_data_dir
    remove_names = json_manager.load(args.remove_names)
    tokenizer_config = args.tokenizer_config
    utterance_size = int(args.utterance_size)
    max_len = int(args.max_len)
    use_multiprocessing = bool(args.use_multiprocessing.lower() == "true")

    setup_data(
        raw_data_dir,
        preprocessed_data_dir,
        model_use_data_dir,
        remove_names,
        tokenizer_config,
        utterance_size,
        max_len,
        use_multiprocessing,
    )
