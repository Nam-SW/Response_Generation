from os.path import abspath, join
from typing import Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


class DataLoader:
    def __init__(
        self,
        data: str,
        contexts_max_len: int,
        response_max_len: int,
        shuffle=True,
        mask_ids=4,
    ):
        self.data = data
        self.contexts_max_len = contexts_max_len
        self.response_max_len = response_max_len
        self.shuffle = shuffle
        self.mask_ids = mask_ids
        self.cls_token = self.str_to_list(self.data.iloc[0, 0])[0]

        self.utterance_size = self.data.shape[1] - 2
        self.on_epoch_end()

    def __len__(self):
        return len(self.data)

    def str_to_list(self, s):
        return list(map(int, s[1:-1].split(", ")))

    def on_epoch_end(self):
        if self.shuffle:
            self.data = self.data.sample(frac=1).reset_index(drop=True)

    def padding(self, x, length):
        return pad_sequences([x], length, padding="post", dtype=np.int32)[0]

    def add_cls(self, x):
        return [self.cls_token] + x

    def MLE_Task(self, row):
        context = self.add_cls(sum(row[: self.utterance_size], []))
        context = self.padding(context, self.contexts_max_len)
        response = self.padding(
            row[self.utterance_size], self.response_max_len - 1
        )
        y = self.padding(row[-1], self.response_max_len - 1)

        return {"context_ids": context, "response_ids": response, "y": y}

    def WOR_Task(self, row):
        shuffle_idx = np.random.choice([*range(self.utterance_size)])
        x = row[shuffle_idx]
        y = self.padding(self.add_cls(row[shuffle_idx]), self.response_max_len)

        np.random.shuffle(x)
        x = self.padding(self.add_cls(x), self.response_max_len)

        return {"context_ids": x, "y": y}

    def UOR_Task(self, row):
        # TODO: implement
        temp = np.zeros(self.contexts_max_len)
        return {"context_ids": temp, "y": temp}

    def MWR_Task(self, row):
        x = row.copy()
        for i in range(self.utterance_size):
            c = np.array(x[i])
            length = len(c)
            mask_count = np.ceil(length * 0.15).astype(np.int32)
            mask_w_idx = np.random.choice(np.arange(length), mask_count)
            c[mask_w_idx] = self.mask_ids
            x[i] = c.tolist()

        x = self.padding(
            self.add_cls(sum(x[: self.utterance_size], [])),
            self.contexts_max_len,
        )
        y = self.padding(
            self.add_cls(sum(row[: self.utterance_size], [])),
            self.contexts_max_len,
        )

        return {"context_ids": x, "y": y}

    def MUR_Task(self, row):
        x = row.copy()
        idx = np.random.choice([*range(self.utterance_size)])
        x[idx] = [self.mask_ids] * len(x[idx])

        x = self.padding(
            self.add_cls(sum(x[: self.utterance_size], [])),
            self.contexts_max_len,
        )
        y = self.padding(
            self.add_cls(sum(row[: self.utterance_size], [])),
            self.contexts_max_len,
        )

        return {"context_ids": x, "y": y}

    def __getitem__(self, i):
        row = self.data.iloc[i].apply(self.str_to_list)
        for i in range(self.utterance_size):
            row[i] = row[i][1:]

        data = dict()
        data["MLE"] = self.MLE_Task(row)
        data["WOR"] = self.WOR_Task(row)
        data["UOR"] = self.UOR_Task(row)
        data["MWR"] = self.MWR_Task(row)
        data["MUR"] = self.MUR_Task(row)

        return data

    def __call__(self):
        for i in range(len(self)):
            yield self.__getitem__(i)

        self.on_epoch_end()


def get_dataloader(
    data_path: str,
    contexts_max_len: int = 128,
    response_max_len: int = 64,
    validation_split: int = 0.1,
    shuffle: bool = True,
    mask_ids: int = 4,
):
    data_path = abspath(data_path)
    data = pd.read_csv(data_path, encoding="utf-8")
    if shuffle:
        data = data.sample(frac=1).reset_index(drop=True)
    std_idx = int(len(data) * validation_split)
    train_data = data.iloc[:-std_idx]
    test_data = data.iloc[-std_idx:]

    train_dataloader = DataLoader(
        train_data, contexts_max_len, response_max_len, shuffle, mask_ids
    )
    test_dataloader = DataLoader(
        test_data, contexts_max_len, response_max_len, shuffle, mask_ids
    )

    return train_dataloader, test_dataloader


def get_tf_data(dataloader: DataLoader, global_batch_size, repeat=True):
    dtypes = {
        "MLE": {
            "context_ids": tf.int32,
            "response_ids": tf.int32,
            "y": tf.int32,
        }
    }
    for key in ["WOR", "UOR", "MWR", "MUR"]:
        dtypes[key] = {"context_ids": tf.int32, "y": tf.int32}
    dataset = tf.data.Dataset.from_generator(dataloader, output_types=dtypes)

    if repeat:
        dataset = dataset.repeat()
    dataset = dataset.batch(global_batch_size)

    return dataset.prefetch(tf.data.experimental.AUTOTUNE)
