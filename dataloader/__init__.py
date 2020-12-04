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
        len_list = np.count_nonzero(batch, axis=-1)
        utterance_idx = [*range(self.utterance_size)]

        shuffled_context = list()
        shuffled_idx = list()
        y = list()
        for i, L in enumerate(len_list):
            U = batch[i].copy()
            shuffle_idx = np.random.choice(utterance_idx)
            u_, l = U[shuffle_idx], L[shuffle_idx]
            nonzero = u_[:l]
            np.random.shuffle(nonzero)

            u_[:l] = nonzero
            shuffled_context.append(U)
            shuffled_idx.append(shuffle_idx)
            y.append(batch[i, shuffle_idx])

        return (tf.constant(shuffled_context), shuffled_idx), tf.constant(y)

    def get_uor_data(self, batch):
        # TODO: implement
        return (None, None)

    def get_mwr_data(self, batch):
        len_list = np.count_nonzero(batch, axis=-1)
        utterance_idx = [*range(self.utterance_size)]
        mask_count = np.ceil(len_list * 0.15).astype(np.int32)

        masked_context = list()
        masked_utterance_idx = list()
        masked_word_idx = list()
        Y = list()
        for i, (L, N) in enumerate(zip(len_list, mask_count)):
            U = batch[i].copy()
            mask_u_idx = np.random.choice(utterance_idx)
            u_, l, n = U[mask_u_idx], L[mask_u_idx], N[mask_u_idx]

            mask_w_idx = np.random.choice(np.arange(l), n)
            u_[mask_w_idx] = self.mask_ids
            y = batch[i, mask_u_idx, mask_w_idx]

            masked_context.append(U)
            masked_utterance_idx.append(mask_u_idx)
            masked_word_idx.append(mask_w_idx)
            Y.append(y)

        return (tf.constant(masked_context), masked_utterance_idx, masked_word_idx), Y

    def get_mur_data(self, batch):
        utterance_idx = [*range(self.utterance_size)]

        masked_context = list()
        masked_utterance_idx = list()
        Y = list()
        for i in range(self.batch_size):
            U = batch[i].copy()
            mask_u_idx = np.random.choice(utterance_idx)
            U[mask_u_idx] = self.mask_ids
            y = batch[i, mask_u_idx]

            masked_context.append(U)
            masked_utterance_idx.append(mask_u_idx)
            Y.append(y)

        return (tf.constant(masked_context), masked_utterance_idx), tf.constant(Y)

    def MLE_Task(self):
        for i in range(len(self)):
            idx = [*range(i * self.batch_size, (i + 1) * self.batch_size)]
            contexts = tf.constant(self.contexts[idx])
            response = tf.constant(self.response[idx])
            Y = tf.constant(self.Y[idx])

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
