import argparse


from utils.JsonManager import JsonManager
from utils.train import train


if __name__ == "__main__":
    json_manager = JsonManager(__file__)

    parser = argparse.ArgumentParser(
        description="build and training a DialogWithAuxility Model"
    )
    parser.add_argument("--data_dir")
    parser.add_argument("--validation_split", default=0.2)
    parser.add_argument("--data_shuffle", default=True)

    parser.add_argument("--model_save_dir", default="model")
    parser.add_argument("--tensorboard_log_dir", default="logs")
    parser.add_argument("--learning_rate", default=0.0001)
    parser.add_argument("--batch_size", default=80)
    parser.add_argument("--epochs", default=5)
    parser.add_argument("--verbose", default=1)
    parser.add_argument("--gpu_count", default=1)
    parser.add_argument("--model_hparams")
    args = parser.parse_args()

    model_hparams = json_manager.load(args.model_hparams)
    train(args, model_hparams)
