from math import ceil
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
        batch_size=32,
        shuffle=True,
        mask_ids=4,
    ):
        self.contexts = contexts
        self.response = response
        self.Y = Y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mask_ids = mask_ids

        self.utterance_size = self.contexts.shape[1]

        self.end_of_epoch()

    def end_of_epoch(self):
        if self.shuffle:
            s = np.arange(len(self.Y))
            np.random.shuffle(s)

            self.contexts = self.contexts[s]
            self.response = self.response[s]
            self.Y = self.Y[s]

    def get_wor_data(self, batch):
        shuffle_idx = np.random.choice([*range(self.utterance_size)], self.batch_size)

        x = np.zeros((self.batch_size, batch.shape[-1]), dtype=np.int32)
        y = np.zeros((self.batch_size, batch.shape[-1]), dtype=np.int32)
        for i, s_i in enumerate(shuffle_idx):
            temp = batch[i, s_i].copy()
            length = np.count_nonzero(temp, axis=0)
            ids = temp[:length]
            np.random.shuffle(ids)
            temp[:length] = ids

            x[i] = temp
            y[i] = batch[i, s_i]

        return x, y

    def get_uor_data(self, batch):
        # TODO: implement
        return (None, None)

    def get_mwr_data(self, batch):
        def masking(x):
            length = np.count_nonzero(x)
            mask_count = np.ceil(length * 0.15).astype(np.int32)
            mask_w_idx = np.random.choice(np.arange(length), mask_count)
            x[mask_w_idx] = self.mask_ids

            return x

        x = np.apply_along_axis(masking, -1, batch.copy())
        y = batch.copy()

        return x, y

    def get_mur_data(self, batch):
        x = batch.copy()
        for i in range(self.batch_size):
            idx = np.random.choice([*range(self.utterance_size)])
            x[i, idx] = self.mask_ids
        y = batch.copy()

        return x, y

    def MLE_Task(self):
        for i in range(len(self)):
            idx = [*range(i * self.batch_size, (i + 1) * self.batch_size)]
            contexts = self.contexts[idx]
            response = self.response[idx]
            Y = self.Y[idx]

            yield (contexts, response), Y

    def Auxiliary_Task(self):
        task_names = ["wor", "uor", "mwr", "mur"]

        for i in range(len(self)):
            idx = [*range(i * self.batch_size, (i + 1) * self.batch_size)]
            contexts = self.contexts[idx]

            wor = self.get_wor_data(contexts)
            uor = self.get_uor_data(contexts)
            mwr = self.get_mwr_data(contexts)
            mur = self.get_mur_data(contexts)

            tasks = [wor, uor, mwr, mur]

            data_dict = {
                name: {
                    "x": data[0],
                    "y": data[1],
                }
                for name, data in zip(task_names, tasks)
            }

            yield data_dict

    def load_data(self):
        for mle, auxiliary in zip(self.MLE_Task(), self.Auxiliary_Task()):
            yield mle, auxiliary

    def __len__(self):
        return ceil(len(self.Y) / self.batch_size)


def get_dataloader(
    data_dir: str,
    validation_split: float,
    batch_size=32,
    shuffle=True,
    mask_ids=4,
):
    data_dir = abspath(data_dir)
    train, test = load_data(data_dir, validation_split)

    train_dataloader = DataLoader(*train, batch_size, shuffle, mask_ids)
    test_dataloader = DataLoader(*test, batch_size, shuffle, mask_ids)

    return train_dataloader, test_dataloader
