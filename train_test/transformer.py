import re

from dataloader.transformer import create_mask, load
from generator import Generator
from losses import mle_loss
from models.MainModels import Transformer
from tftrainer import TrainArgument, Trainer
from transformers import AutoTokenizer


def processing(text):
    if re.match("[ㄱ-ㅎ]+", text):
        text_set = set(text)
        if len({"ㅋ", "ㅌ", "ㄴ"} - text_set) < 2:
            text = "ㅋ" * len(text)

        elif {"ㅎ", "ㄹ"} == text_set:
            text = "ㅎ" * len(text)

    text = text.replace("\n", " ").strip()
    text = re.sub(r" {2,}", r" ", text)
    text = re.sub(r"(.{8,}?)\1+", r"\1", text)
    text = re.sub(r"[^ ㄱ-ㅎㅏ-ㅣ가-힣A-Za-z0-9~!@#$%\^\&*\(\)_+-=\[\]{},./<>?]", r"", text)
    text = re.sub(r"http.+", r"<url>", text)
    text = text.replace("[NAME]", "<name>").replace("[URL]", "<url>").strip()

    return text


def train(cfg):
    cfg = cfg.TRANSFORMER
    tokenizer = AutoTokenizer.from_pretrained(cfg.ETC.tokenizer_path)
    args = TrainArgument(**cfg.TRAINARGS)

    train_dataset, eval_dataset = load(tokenizer=tokenizer, **cfg.DATASETS)

    with args.strategy.scope():
        model = Transformer(vocab_size=tokenizer.vocab_size, **cfg.MODEL)

    trainer = Trainer(
        model,
        args,
        train_dataset,
        loss_function=mle_loss,
        eval_dataset=eval_dataset,
        data_collator=create_mask,
    )

    trainer.train()

    model.save(cfg.ETC.output_dir)


def test(cfg):
    cfg = cfg.TRANSFORMER
    tokenizer = AutoTokenizer.from_pretrained(cfg.ETC.tokenizer_path)
    model = Transformer.load(cfg.ETC.output_dir)

#     user_token = "<user1>"
#     model_token_id = tokenizer.convert_tokens_to_ids("<user2>")

    g = Generator(model, tokenizer)

    talk_log = []

    while True:
        text_list = []
        while True:
            text = input("Me: ")
            if text == "":
                break
            elif text == "종료":
                exit()
            text_list.append(text)

        text = processing(" ".join(text_list))
        talk_log.append(text)
        input_text = talk_log[-3:]

        # pred = g.generate_greedy(input_text, temperature=1.5)
        pred = g.generate_random_sampling(input_text, temperature=1.5, top_k=20)
        # pred = g.generate_beam(
        #     input_text=input_text,
        #     num_beams=3,
        #     max_length=128,
        #     temperature=1.5,
        # )
        pred = pred[1:]
        if pred[-1] == tokenizer.eos_token_id:
            pred = pred[:-1]

        result = tokenizer.decode(pred)
        print("BOT: " + result)
        talk_log.append(processing(result))
