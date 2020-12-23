import argparse

from dataloader.preprocessing import setup_data
from utils.JsonManager import JsonManager


if __name__ == "__main__":
    json_manager = JsonManager(__file__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_dir")
    parser.add_argument("--preprocessed_data_dir")
    parser.add_argument("--model_use_data_dir")
    parser.add_argument("--remove_names")
    parser.add_argument("--tokenizer_config")
    parser.add_argument("--utterance_size", type=int, default=4)
    parser.add_argument("--max_len", type=int, default=100)
    parser.add_argument("--use_multiprocessing", type=bool, default=True)

    args = parser.parse_args()

    raw_data_dir = args.raw_data_dir
    preprocessed_data_dir = args.preprocessed_data_dir
    model_use_data_dir = args.model_use_data_dir
    remove_names = json_manager.load(args.remove_names)
    tokenizer_config = args.tokenizer_config
    utterance_size = args.utterance_size
    max_len = args.max_len
    use_multiprocessing = args.use_multiprocessing

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
