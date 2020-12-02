import argparse

from utils import JsonManager
from dataloader import get_dataloader


if __name__ == "__main__":
    json_manager = JsonManager(__file__)

    parser = argparse.ArgumentParser(
        description="build and training a DialogWithAuxility Model"
    )
    parser.add_argument("--data_dir")
    parser.add_argument("--validation_split", default=0.2)
    parser.add_argument("--data_shuffle", default=True)

    parser.add_argument("--model_save_dir", default="model")
    parser.add_argument("--learning_rate", default=0.0001)
    parser.add_argument("--batch_size", default=80)
    parser.add_argument("--epochs", default=5)
    parser.add_argument("--verbose", default=1)
    parser.add_argument("--model_hparams")
    args = parser.parse_args()

    data_dir = args.data_dir
    validation_split = float(args.validation_split)
    data_shuffle = (args.data_shuffle.lower()) == "true"

    model_save_dir = args.model_save_dir
    learning_rate = float(args.learning_rate)
    batch_size = int(args.batch_size)
    epochs = int(args.epochs)
    verbose = int(args.verbose)
    model_hparams = json_manager.load(args.model_hparams)

    train_dataloader, test_dataloader = get_dataloader(
        data_dir,
        validation_split,
        batch_size,
        data_shuffle,
    )

    # TODO: training implement
