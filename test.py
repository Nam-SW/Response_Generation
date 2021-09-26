import os
import sys

import yaml
from prodict import Prodict

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from transformers import AutoTokenizer

from dataloader.poly_ecoder import load
from models.MainModels import PolyEncoder

argv = sys.argv


def main():
    with open("config.yml") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        cfg = Prodict.from_dict(cfg)

    cfg = cfg.POLYENCODER

    tokenizer = AutoTokenizer.from_pretrained(cfg.ETC.tokenizer_path)

    train_dataset, eval_dataset = load(tokenizer=tokenizer, **cfg.DATASETS)
    model = PolyEncoder(**cfg.MODEL)

    data = train_dataset[:4]
    for k, v in data.items():
        data[k] = tf.constant(data[k])

    pred = model(**data)
    print(pred)


main()
