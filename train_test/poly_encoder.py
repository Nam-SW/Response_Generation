from dataloader.poly_ecoder import load
from models.MainModels import PolyEncoder
from trainer import TrainArgument, Trainer
from transformers import AutoTokenizer


def train(cfg):
    cfg = cfg.POLYENCODER
    tokenizer = AutoTokenizer.from_pretrained(cfg.ETC.tokenizer_path)

    train_dataset, eval_dataset = load(tokenizer=tokenizer, **cfg.DATASETS)

    args = TrainArgument(**cfg.TRAINARGS)

    with args.strategy.scope():
        model = PolyEncoder(**cfg.MODEL)

    trainer = Trainer(
        model,
        args,
        train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()

    model.save(cfg.ETC.output_dir)


def test(cfg):
    print("test poly encoder")
