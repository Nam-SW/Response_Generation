from os.path import abspath

import numpy as np
import tensorflow as tf

from .preprocessing import load_data


class DataLoader:
    def __init__(
        self,
        contexts: np.ndarray,
        response: np.ndarray,
        Y: np.ndarray,
        shuffle=True,
        mask_ids=4,
    ):
        self.contexts = contexts
        self.response = response
        self.Y = Y
        self.shuffle = shuffle
        self.mask_ids = mask_ids

        self.utterance_size = self.contexts.shape[1]

    def start_of_epoch(self):
        if self.shuffle:
            s = np.arange(len(self.Y))
            np.random.shuffle(s)

            self.contexts = self.contexts[s]
            self.response = self.response[s]
            self.Y = self.Y[s]

    def get_wor_data(self, batch):
        shuffle_idx = np.random.choice([*range(self.utterance_size)])

        x = batch[shuffle_idx].copy()
        y = batch[shuffle_idx].copy()

        length = np.count_nonzero(x)
        ids = x[:length]
        np.random.shuffle(ids)
        x[:length] = ids

        return x.tolist(), y.tolist()

    def get_uor_data(self, batch):
        # TODO: implement
        return [0], [0]

    def get_mwr_data(self, batch):
        x = batch.copy()
        for i in range(self.utterance_size):
            length = np.count_nonzero(x[i])
            mask_count = np.ceil(length * 0.15).astype(np.int32)
            mask_w_idx = np.random.choice(np.arange(length), mask_count)
            x[i, mask_w_idx] = self.mask_ids
        y = batch.copy()

        return x.tolist(), y.tolist()

    def get_mur_data(self, batch):
        x = batch.copy()
        idx = np.random.choice([*range(self.utterance_size)])
        x[idx] = self.mask_ids
        y = batch.copy()

        return x.tolist(), y.tolist()

    def MLE_Task(self):
        for i in range(len(self)):
            # idx = [*range(i * self.batch_size, (i + 1) * self.batch_size)]
            # contexts = self.contexts[idx]
            # response = self.response[idx]
            # Y = self.Y[idx]
            contexts = self.contexts[i]
            response = self.response[i]
            Y = self.Y[i]

            yield (contexts.tolist(), response.tolist()), Y

    def Auxiliary_Task(self):
        for i in range(len(self)):
            # idx = [*range(i * self.batch_size, (i + 1) * self.batch_size)]
            # contexts = self.contexts[idx]
            contexts = self.contexts[i]

            wor = self.get_wor_data(contexts)
            uor = self.get_uor_data(contexts)
            mwr = self.get_mwr_data(contexts)
            mur = self.get_mur_data(contexts)

            tasks = (wor, uor, mwr, mur)

            yield tasks

    def load_data(self):
        self.start_of_epoch()

        for mle, auxiliary in zip(self.MLE_Task(), self.Auxiliary_Task()):
            yield mle, auxiliary

    def __len__(self):
        return len(self.Y)


def get_dataloader(
    data_dir: str,
    validation_split: float,
    # batch_size=32,
    shuffle=True,
    mask_ids=4,
):
    data_dir = abspath(data_dir)
    train, test = load_data(data_dir, validation_split)

    train_dataloader = DataLoader(*train, shuffle, mask_ids)
    test_dataloader = DataLoader(*test, shuffle, mask_ids)

    return train_dataloader, test_dataloader


def get_tf_data(dataloader: DataLoader, global_batch_size, repeat=True):
    dtypes = (
        ((tf.int32, tf.int32), tf.int32),
        tuple([(tf.int32, tf.int32) for _ in range(4)]),
    )
    dataset = tf.data.Dataset.from_generator(dataloader, output_types=dtypes)

    if repeat:
        dataset = dataset.repeat()
    dataset = dataset.batch(global_batch_size)

    return dataset
