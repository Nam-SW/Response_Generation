import argparse

from tensorflow.distribute import MirroredStrategy

from dataloader import get_dataloader
from utils.JsonManager import JsonManager
from utils.TrainManager import TrainManager


def main(args, model_hparams):
    train_dataloader, test_dataloader = get_dataloader(
        args.data_dir,
        float(args.validation_split),
        int(args.batch_size),
        (args.data_shuffle.lower()) == "true",
    )

    strategy = MirroredStrategy()

    with strategy.scope():
        trainer = TrainManager(model_hparams)
        trainer.compile(float(args.learning_rate))
        trainer.train(
            train_dataloader,
            test_dataloader,
            model_save_dir=args.model_save_dir,
            tensorboard_log_dir=args.tensorboard_log_dir,
            epochs=int(args.epochs),
            verbose=int(args.verbose),
        )


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
    parser.add_argument("--model_hparams")
    args = parser.parse_args()

    model_hparams = json_manager.load(args.model_hparams)
    main(args, model_hparams)
