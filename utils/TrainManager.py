import os
from datetime import datetime as dt
from typing import Dict, Optional, Tuple

import tensorflow as tf
from tqdm import tqdm

from modeling.model import DialogWithAuxility


class TrainManager:
    def __init__(self, model_hparams: Dict, **kwargs):
        self.model = DialogWithAuxility(**model_hparams)
        self.optimizer = None

        self.metrics_name = [
            "adversarial_loss",
            "MLE_loss",
            "Auxiliary_loss",
            "WOR_loss",
            # "UOR_loss",  # TODO: implement
            "MWR_loss",
            "MUR_loss",
        ]

        self.loss = tf.keras.losses.SparseCategoricalCrossentropy()

        self.train_metrics = {}
        self.valid_metrics = {}

        self.init_metrics()

    def init_metrics(self):
        for name in self.metrics_name:
            self.train_metrics[name] = tf.keras.metrics.Mean()
            self.valid_metrics[name] = tf.keras.metrics.Mean()

    def init_training_hparams(self, BatchPerEpoch, epochs, T2=30):
        self.alpha = 1.0
        self.N = BatchPerEpoch
        self.T1 = epochs * self.N
        self.T2 = T2
        self.d = self.alpha / (self.T2 * self.N)

    def compile(self, learning_late: 0.05, **kwargs):
        self.optimizer = tf.keras.optimizers.Adagrad(learning_late, **kwargs)

    def train(
        self,
        train_dataloader,
        test_dataloader,
        model_save_dir: str,
        tensorboard_log_dir: Optional[str],
        epochs: int = 5,
        verbose: int = 1,
    ):
        assert self.optimizer is not None, "model must be compiled before training."

        self.init_training_hparams(len(train_dataloader), epochs)

        model_save_dir = os.path.abspath(model_save_dir)
        if os.path.isdir(model_save_dir):
            os.mkdir(model_save_dir)

        if tensorboard_log_dir:
            tensorboard_log_dir = os.path.abspath(tensorboard_log_dir)

            current_time = dt.now().strftime("%Y%m%d-%H%M%S")

            train_log_dir = os.path.join(tensorboard_log_dir, current_time, "train")
            valid_log_dir = os.path.join(tensorboard_log_dir, current_time, "test")

            self.train_summary_writter = tf.summary.create_file_writer(train_log_dir)
            self.valid_summary_writter = tf.summary.create_file_writer(valid_log_dir)

        for epoch in range(epochs):
            if verbose:
                print(f"{epoch + 1}/{epochs} epochs...")

            train_data_generator = train_dataloader.load_data()
            valid_data_generator = test_dataloader.load_data()
            if verbose:
                train_data_generator = tqdm(train_data_generator)
                valid_data_generator = tqdm(valid_data_generator)

            # TODO: 여기 어딘가에서 에러가 남. 왜지
            for batch in train_data_generator:
                self._train_batch(batch, self.alpha > 0, training=True)

            for batch in valid_data_generator:
                self._train_batch(batch, self.alpha > 0, training=False)

            if verbose:
                for key, value in self.train_metrics.items():
                    print(f"{key}: {value.result()}", end="\t")

                print()

                for key, value in self.valid_metrics.items():
                    print(f"{key}: {value.result()}", end="\t")

                print()

            self.model.save_weights(
                os.path.join(model_save_dir), f"model_weight_{epoch+1: 02d}"
            )
            if tensorboard_log_dir:
                self._write_on_tensorboard(epoch)

            self.alpha = max(0, self.alpha - self.d)

            train_dataloader.end_of_epoch()
            test_dataloader.end_of_epoch()

    def _train_batch(
        self,
        data: Tuple,
        compute_auxiliary: bool = False,
        training: bool = True,
    ):
        mle_data, auxiliary_data = data

        mle_x, mle_y = mle_data

        auxiliary_loss = 0
        WOR_loss = 0
        # UOR_loss = 0
        MWR_loss = 0
        MUR_loss = 0

        with tf.GradientTape() as tape:
            mle_pred = self.model(mle_x, return_all_sequences=True, training=training)
            mle_loss = self.loss(mle_y, mle_pred)

            if compute_auxiliary:
                WOR_pred = self.model.call_word_order_recovery(
                    *auxiliary_data["wor"]["x"]
                )
                # UOR_pred = self.model.call_utterance_order_recovery(
                #     *auxiliary_data["uor"]["x"]
                # )
                MWR_pred = self.model.call_masked_word_recovery(
                    *auxiliary_data["mwr"]["x"]
                )
                MUR_pred = self.model.call_masked_utterance_recovery(
                    *auxiliary_data["mur"]["x"]
                )

                WOR_loss += self.loss(auxiliary_data["wor"]["y"], WOR_pred)
                # UOR_loss += self.loss(auxiliary_data["uor"]["y"], UOR_pred)
                MWR_loss += tf.reduce_mean(
                    [
                        self.loss(pred, MWR_pred[i])
                        for i, pred in enumerate(auxiliary_data["mwr"]["y"])
                    ]
                )
                MUR_loss += self.loss(auxiliary_data["mur"]["y"], MUR_pred)

                # auxiliary_loss += WOR_loss + UOR_loss + MWR_loss + MUR_loss
                auxiliary_loss += WOR_loss + MWR_loss + MUR_loss

            adversarial_loss = mle_loss + self.alpha * auxiliary_loss

        if training:
            gradient = tape.gradient(adversarial_loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(
                zip(gradient, self.model.trainable_variables)
            )

            self.train_metrics["adversarial_loss"].update_state(adversarial_loss)
            self.train_metrics["MLE_loss"].update_state(mle_loss)
            self.train_metrics["Auxiliary_loss"].update_state(auxiliary_loss)
            self.train_metrics["WOR_loss"].update_state(WOR_loss)
            # self.train_metrics["UOR_loss"].update_state(UOR_loss)
            self.train_metrics["MWR_loss"].update_state(MWR_loss)
            self.train_metrics["MUR_loss"].update_state(MUR_loss)

        else:
            self.valid_metrics["adversarial_loss"].update_state(adversarial_loss)
            self.valid_metrics["MLE_loss"].update_state(mle_loss)
            self.valid_metrics["Auxiliary_loss"].update_state(auxiliary_loss)
            self.valid_metrics["WOR_loss"].update_state(WOR_loss)
            # self.valid_metrics["UOR_loss"].update_state(UOR_loss)
            self.valid_metrics["MWR_loss"].update_state(MWR_loss)
            self.valid_metrics["MUR_loss"].update_state(MUR_loss)

        return None

    def _write_on_tensorboard(self, epoch):
        with self.train_summary_writter.as_default():
            for key, value in self.train_metrics.items():
                tf.summary.scalar(key, value.result(), epoch)

        with self.valid_summary_writter.as_default():
            for key, value in self.valid_metrics.items():
                tf.summary.scalar(key, value.result(), epoch)
