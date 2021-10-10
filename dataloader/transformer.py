from os import environ
from os.path import abspath, splitext
from typing import Optional

import numpy as np
import tensorflow as tf
from datasets import load_dataset, logging, set_caching_enabled
from tensorflow.keras.preprocessing.sequence import pad_sequences


logging.set_verbosity(logging.WARN)


def load(
    tokenizer,
    seq_len: int,
    window: int,
    train_data_path: str,
    eval_data_path: Optional[str] = None,
    train_test_split: Optional[float] = None,
    worker: int = 1,
    batch_size: int = 1000,
    shuffle_seed: Optional[int] = None,
):
    def _tokenize(sample):
        sample["tokenized"] = tokenizer(sample["content"])["input_ids"]
        return sample

    def _grouping(sample):
        input_ids = []
        labels = []

        contents = [[] for _ in range(window - 1)] + sample["tokenized"]
        talk_ids = [sample["talk_id"][0] for _ in range(window - 1)] + sample["talk_id"]

        s, e = 0, window
        now_talk_id = talk_ids[0]

        while len(contents) > e:
            talk_id = talk_ids[e]

            if now_talk_id != talk_id:
                contents = [[] for _ in range(window - 1)] + contents[s + window :]
                talk_ids = [talk_id for _ in range(window - 1)] + talk_ids[s + window :]
                s, e = 0, window
                now_talk_id = talk_id
                continue

            input_ids.append(contents[s:e])
            labels.append(contents[e])

            s += 1
            e += 1

        return {"input_ids": input_ids, "labels": labels}

    def _padding_and_set_attention_mask(sample):
        def _padding(data):
            return pad_sequences(
                data,
                seq_len,
                padding="post",
                truncating="post",
            )

        input_ids = sum(sample["input_ids"], [])
        input_ids = np.reshape(_padding(input_ids), (-1, window, seq_len))
        attention_mask = (input_ids != 0).astype(np.int32)

        decoder_input_ids = [[tokenizer.bos_token_id] + s for s in sample["labels"]]
        decoder_input_ids = _padding(decoder_input_ids)
        decoder_attention_mask = (decoder_input_ids != 0).astype(np.int32)

        labels = [s + [tokenizer.eos_token_id] for s in sample["labels"]]
        labels = _padding(labels)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "labels": labels,
        }
    
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
    print('load')

    data = data.map(
        _tokenize,
        batched=True,
        batch_size=batch_size,
        num_proc=worker,
    )
    print('tokenizer')

    data = data.map(
        _grouping,
        batched=True,
        batch_size=batch_size,
        num_proc=worker,
        remove_columns=data["train"].column_names,
    )
    print('prouping')

    data = data.map(
        _padding_and_set_attention_mask,
        batched=True,
        batch_size=batch_size,
        num_proc=worker,
    )
    print('padding')

    if shuffle_seed:
        data = data.shuffle(seed=shuffle_seed)

    return data["train"], (data["test"] if is_eval else None)


def create_look_ahead_mask_collator(data):
    padding_mask = 1 - tf.cast(tf.equal(data["decoder_input_ids"], 0), tf.int32)
    size = tf.shape(padding_mask)[-1]

    look_ahead_mask = tf.linalg.band_part(tf.ones((size, size)), -1, 0)

    data["decoder_look_ahead_mask"] = tf.minimum(
        padding_mask[:, tf.newaxis, tf.newaxis, :],
        tf.cast(look_ahead_mask, tf.int32),
    )
    return data
