from os.path import abspath, splitext
from typing import Optional

import numpy as np
import tensorflow as tf
from datasets import load_dataset, logging
from tensorflow.keras.preprocessing.sequence import pad_sequences

logging.set_verbosity(logging.ERROR)


def to_tf_dataset(dataset):
    signature = (
        {
            "input_ids": tf.TensorSpec(shape=None, dtype="int32"),
            "decoder_input_ids": tf.TensorSpec(shape=None, dtype="int32"),
        },
        tf.TensorSpec(shape=None, dtype="int32"),
    )

    def _gen():
        for batch in dataset:
            y = batch.pop("labels")
            yield batch, y

    tf_dataset = tf.data.Dataset.from_generator(_gen, output_signature=signature)

    tf_dataset = tf_dataset.apply(tf.data.experimental.assert_cardinality(len(dataset)))

    return tf_dataset


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
#     def _tokenize(sample):
#         sample["tokenized"] = tokenizer(sample["content"])["input_ids"]
#         return sample

    def _grouping(sample):
        def _padding(data):
            return pad_sequences(
                data,
                seq_len,
                padding="post",
                truncating="post",
            )

        bos = [tokenizer.bos_token_id]
        eos = [tokenizer.eos_token_id]
        sample["tokenized"] = [s[1:] for s in sample["tokenized"]]

        input_ids = []
        decoder_input_ids = []
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
            
            input_id = []
            for c in contents[s:e]:
                if input_id:
                    input_id.append(tokenizer.sep_token_id)
                input_id += c[:seq_len // 2]

            input_ids.append(input_id)
            decoder_input_ids.append(bos + contents[e])
            labels.append(contents[e] + eos)

            s += 1
            e += 1

        return {
            "input_ids": _padding(input_ids),
            "decoder_input_ids": _padding(decoder_input_ids),
            "labels": _padding(labels),
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

#     data = data.map(
#         _tokenize,
#         batched=True,
#         batch_size=batch_size,
#         num_proc=worker,
#     )

    data = data.map(
        _grouping,
        batched=True,
        batch_size=batch_size,
        num_proc=worker,
        remove_columns=data["train"].column_names,
    )

    if shuffle_seed:
        data = data.shuffle(seed=shuffle_seed)

    return (
        to_tf_dataset(data["train"]),
        (to_tf_dataset(data["test"]) if is_eval else None),
    )


def create_mask(x, y):
    x["attention_mask"] = tf.cast(tf.not_equal(x["input_ids"], 0), tf.int32)
    x["decoder_attention_mask"] = tf.cast(
        tf.not_equal(x["decoder_input_ids"], 0), tf.int32
    )
    return (x, y)
