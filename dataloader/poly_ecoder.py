from os.path import abspath, splitext
from typing import Optional

import numpy as np
import tensorflow as tf
from datasets import load_dataset, logging

logging.set_verbosity_warning()


def load(
    tokenizer,
    seq_len: int,
    train_data_path: str,
    eval_data_path: Optional[str] = None,
    train_test_split: Optional[float] = None,
    worker: int = 1,
    batch_size: int = 1000,
    shuffle_seed: Optional[int] = None,
):
    def _split_decoder_data(sample):
        sample["encoder_input"] = sample["encoder_input"].replace("<sep>", " ")
        sample["decoder_input"] = sample["decoder_input"].replace("<sep>", " ")
        return sample

    def _tokenize_function(sample):
        tokenized = dict()

        e = tokenizer(
            sample["encoder_input"],
            max_length=seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )
        tokenized["context_input_ids"] = e["input_ids"]
        tokenized["context_mask"] = e["attention_mask"]

        d = tokenizer(
            sample["decoder_input"],
            max_length=seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )
        tokenized["candidate_input_ids"] = d["input_ids"][:, None, :]
        # tokenized["decoder_padding_mask"] = e["attention_mask"][:, None, :]

        tokenized["labels"] = np.ones(tokenized["candidate_input_ids"].shape[:-1])

        return tokenized

    train_data_path = abspath(train_data_path)
    is_eval = False
    _, extention = splitext(train_data_path)

    datafiles = {"train": train_data_path}
    if eval_data_path is not None:
        assert (
            train_test_split is None
        ), "Only one of eval_data_path and train_test_split must be entered."
        datafiles["test"] = abspath(eval_data_path)
        is_eval = True

    if train_test_split is not None:
        assert (
            0.0 < train_test_split < 1.0
        ), "train_test_split must be a value between 0 and 1"
        train_test_split = int(train_test_split * 100)
        train_test_split = {
            "train": f"train[:{train_test_split}%]",
            "test": f"train[{train_test_split}%:]",
        }
        is_eval = True

    data = load_dataset(
        extention.replace(".", ""), data_files=datafiles, split=train_test_split
    )
    if shuffle_seed:
        data = data.shuffle(seed=shuffle_seed)

    data = data.map(
        _split_decoder_data,
        num_proc=worker,
    )

    data = data.map(
        _tokenize_function,
        batched=True,
        batch_size=batch_size,
        num_proc=worker,
        remove_columns=data["train"].column_names,
    )

    return data["train"], (data["test"] if is_eval else None)


def create_look_ahead_mask_collator(data):
    padding_mask = 1 - tf.cast(tf.equal(data["decoder_input_ids"], 0), tf.int64)
    size = tf.shape(padding_mask)[-1]

    look_ahead_mask = tf.linalg.band_part(tf.ones((size, size)), -1, 0)

    data["decoder_look_ahead_mask"] = tf.minimum(
        padding_mask[:, tf.newaxis, tf.newaxis, :],
        tf.cast(look_ahead_mask, tf.int64),
    )
    return data
