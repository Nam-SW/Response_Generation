import os
import sys

import yaml
from prodict import Prodict

os.environ["HF_DATASETS_CACHE"] = "/mnt/subdisk/huggingface"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["DATASETS_VERBOSITY"] = "error"

from train_test import poly_encoder, transformer

argv = sys.argv


def main():
    with open("config.yml") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        cfg = Prodict.from_dict(cfg)

    action, model_name = argv[1:]
    model_name = model_name.replace("-", "_").lower()

    if action == "train":
        if model_name == "poly_encoder":
            poly_encoder.train(cfg)
        elif model_name == "transformer":
            transformer.train(cfg)

    elif action == "test":
        if model_name == "poly_encoder":
            poly_encoder.test(cfg)
        elif model_name == "transformer":
            transformer.test(cfg)


if __name__ == "__main__":
    main()
