import argparse


from utils.JsonManager import JsonManager
from utils.train import train


if __name__ == "__main__":
    json_manager = JsonManager(__file__)

    parser = argparse.ArgumentParser(
        description="build and training a DialogWithAuxility Model"
    )
    parser.add_argument("--data_path")
    parser.add_argument("--validation_split", type=float, default=0.2)
    parser.add_argument("--data_shuffle", type=str, default="True")
    parser.add_argument("--contexts_max_len", type=int, default=128)
    parser.add_argument("--response_max_len", type=int, default=64)

    parser.add_argument("--model_save_dir", default="model")
    parser.add_argument("--tensorboard_log_dir", default=None)
    parser.add_argument("--warmup_steps", type=int, default=5000)
    # parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--batch_size", type=int, default=80)
    parser.add_argument("--global_max_step", type=int, default=50000)
    parser.add_argument("--validation_step", type=int, default=1000)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--gpu_count", type=int, default=1)
    parser.add_argument("--model_hparams")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--load_latest", type=str, default="False")
    args = parser.parse_args()

    model_hparams = json_manager.load(args.model_hparams)
    train(args, model_hparams)
