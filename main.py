import warnings
from math import exp
from os.path import splitext

import hydra
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

warnings.filterwarnings(action="ignore")


def load(
    data_path: str,
    tokenizer,
    seq_len: int,
    target_column: str = "text",
    worker: int = 1,
):
    def _tokenize_function(text):
        tokenized = tokenizer(text[target_column])
        return tokenized

    def _group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // seq_len) * seq_len

        result = {
            k: [t[i : i + seq_len] for i in range(0, total_length, seq_len)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    _, extention = splitext(data_path)
    data = load_dataset(extention.replace(".", ""), data_files=data_path)

    tokenized_datasets = data.map(
        _tokenize_function,
        batched=True,
        num_proc=worker,
        remove_columns=data["train"].column_names,
    )

    lm_datasets = tokenized_datasets.map(_group_texts, batched=True, num_proc=worker)

    return lm_datasets["train"]


@hydra.main(config_name="config.yaml")
def main(cfg):
    # tokenizer 로드
    tokenizer = GPT2TokenizerFast.from_pretrained(cfg.PATH.tokenizer)
    do_eval = None if cfg.PATH.eval_path is None else True

    # 데이터 로드
    train_dataset = load(
        data_path=cfg.PATH.train_path,
        tokenizer=tokenizer,
        **cfg.PROCESSINGARGS,
    )
    eval_dataset = (
        load(
            data_path=cfg.PATH.eval_path,
            tokenizer=tokenizer,
            **cfg.PROCESSINGARGS,
        )
        if do_eval
        else None
    )

    # 모델 로드
    model = AutoModelForCausalLM.from_pretrained(cfg.PATH.model_name)

    # 학습 arg 세팅
    args = TrainingArguments(
        do_train=True,
        do_eval=do_eval,
        logging_dir=cfg.PATH.logging_dir,
        output_dir=cfg.PATH.checkpoint_dir,
        **cfg.TRAININGARGS,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=None,
        data_collator=default_data_collator,
    )

    # 학습 시작
    train_result = trainer.train()

    # 학습 결과 저장
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # 모델 저장
    trainer.save_model(cfg.PATH.output_dir)

    if do_eval:
        # 검증 시작
        metrics = trainer.evaluate()

        # 검증 결과 저장
        metrics["eval_samples"] = len(eval_dataset)
        perplexity = exp(metrics["eval_loss"])
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
