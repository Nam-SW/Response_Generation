import os
from datetime import datetime as dt
from typing import Optional, Tuple

import tensorflow as tf


class TrainManager:
    def __init__(self, model, gpu_count, batch_size):
        self.model = model
        self.gpu_count = gpu_count
        self.batch_size = batch_size
        self.global_batch_size = self.gpu_count * self.batch_size
        self.use_tensorboard = False

        self.metrics_name = [
            "adversarial_loss",
            "MLE_loss",
            "Auxiliary_loss",
            "WOR_loss",
            # "UOR_loss",  # TODO: implement
            "MWR_loss",
            "MUR_loss",
        ]

        self.set_loss()
        self.init_metrics()

    def set_loss(self):
        self.sparse_cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy()

        def MLE_loss(y_true, y_pred):
            # loss = self.sparse_cross_entropy(y_true, y_pred)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
            return tf.nn.compute_average_loss(
                loss, global_batch_size=self.global_batch_size
            )

        def mcr_loss(y_true, y_pred):
            idx = y_true == 4
            y_true_mask = y_true[idx]
            y_pred_mask = y_pred[idx]
            # loss = self.sparse_cross_entropy_loss(y_true_mask, y_pred_mask)
            loss = tf.keras.losses.sparse_categorical_crossentropy(
                y_true_mask, y_pred_mask
            )
            return tf.nn.compute_average_loss(
                loss, global_batch_size=self.global_batch_size
            )

        self.MLE_loss = MLE_loss
        self.MCR_loss = mcr_loss

    def init_metrics(self):
        self.train_metrics = {}
        self.valid_metrics = {}

        for name in self.metrics_name:
            self.train_metrics[name] = tf.keras.metrics.Mean("train_" + name)
            self.valid_metrics[name] = tf.keras.metrics.Mean("valid_" + name)

    def init_training_hparams(self, BatchPerEpoch, epochs, T2=30):
        self.alpha = 1.0
        self.N = BatchPerEpoch
        self.T2 = T2
        self.d = self.alpha / (self.T2 * self.N)
        self.train_global_step = 0
        self.valid_global_step = 0

    def compile(self, optimizer):
        self.optimizer = optimizer

    def setup_tensorboard(self, tensorboard_log_dir):
        current_time = dt.now().strftime("%Y%m%d-%H%M%S")

        train_log_dir = os.path.join(tensorboard_log_dir, current_time, "train")
        valid_log_dir = os.path.join(tensorboard_log_dir, current_time, "test")

        self.train_summary_writter = tf.summary.create_file_writer(train_log_dir)
        self.valid_summary_writter = tf.summary.create_file_writer(valid_log_dir)

    def train(
        self,
        train_dataloader,
        test_dataloader,
        strategy,
        BatchPerEpoch: int,
        model_save_dir: str,
        tensorboard_log_dir: Optional[str],
        epochs: int = 5,
        verbose: int = 1,
    ):
        assert self.optimizer is not None, "model must be compiled before training."

        self.strategy = strategy
        self.init_training_hparams(BatchPerEpoch, epochs)

        model_save_dir = os.path.abspath(model_save_dir)
        if os.path.isdir(model_save_dir):
            os.mkdir(model_save_dir)

        if tensorboard_log_dir:
            tensorboard_log_dir = os.path.abspath(tensorboard_log_dir)
            self.setup_tensorboard(tensorboard_log_dir)
            self.use_tensorboard = True

        # print(self.use_tensorboard)
        for epoch in range(epochs):
            if verbose:
                print(f"{epoch + 1}/{epochs} epochs...")

            for batch in train_dataloader:
                # print("in train")
                self.distributed_train_batch(batch, self.alpha > 0, training=True)
                self._write_on_tensorboard_train()
                self.train_global_step += 1

            for batch in test_dataloader:
                self.distributed_train_batch(batch, self.alpha > 0, training=False)
                self._write_on_tensorboard_valid()
                self.valid_global_step += 1

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

            self.alpha = max(0, self.alpha - self.d)

    def _write_on_tensorboard_train(self):
        if not self.use_tensorboard:
            return
        with self.train_summary_writter.as_default():
            for value in self.train_metrics.values():
                tf.summary.scalar(value.name, value.result(), self.train_global_step)
                value.reset_states()
        # print("write_train")

    def _write_on_tensorboard_valid(self):
        if not self.use_tensorboard:
            return
        with self.valid_summary_writter.as_default():
            for value in self.valid_metrics.values():
                tf.summary.scalar(value.name, value.result(), self.valid_global_step)
                value.reset_states()
            # print("write_valid")

    @tf.function
    def distributed_train_batch(self, data, compute_auxiliary, training):
        loss = self.strategy.experimental_run_v2(
            self._train_batch, args=(data, compute_auxiliary, training)
        )
        return loss

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
            mle_pred = self.model(
                mle_x, return_all_sequences=True, task="MLE", training=training
            )
            mle_loss = self.MLE_loss(mle_y, mle_pred)

            if compute_auxiliary:
                wor, uor, mwr, mur = auxiliary_data

                WOR_pred = self.model(wor[0], task="WOR", training=training)
                # UOR_pred = self.model(uor[0], task="UOR", training=training)
                MWR_pred = self.model(mwr[0], task="MCR", training=training)
                MUR_pred = self.model(mwr[0], task="MCR", training=training)

                WOR_loss += self.MLE_loss(wor[1], WOR_pred)
                # UOR_loss += self.loss(uor[1], UOR_pred)
                MWR_loss += self.MCR_loss(mwr[1], MWR_pred)
                MUR_loss += self.MCR_loss(mwr[1], MUR_pred)

                # auxiliary_loss += WOR_loss + UOR_loss + MWR_loss + MUR_loss
                auxiliary_loss += WOR_loss + MWR_loss + MUR_loss

            adversarial_loss = mle_loss + self.alpha * auxiliary_loss

        if training:
            gradient = tape.gradient(adversarial_loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(
                zip(gradient, self.model.trainable_variables)
            )

            self.train_metrics["adversarial_loss"](adversarial_loss)
            self.train_metrics["MLE_loss"](mle_loss)
            self.train_metrics["Auxiliary_loss"](auxiliary_loss)
            self.train_metrics["WOR_loss"](WOR_loss)
            # self.train_metrics["UOR_loss"](UOR_loss)
            self.train_metrics["MWR_loss"](MWR_loss)
            self.train_metrics["MUR_loss"](MUR_loss)

        else:
            self.valid_metrics["adversarial_loss"](adversarial_loss)
            self.valid_metrics["MLE_loss"](mle_loss)
            self.valid_metrics["Auxiliary_loss"](auxiliary_loss)
            self.valid_metrics["WOR_loss"](WOR_loss)
            # self.valid_metrics["UOR_loss"](UOR_loss)
            self.valid_metrics["MWR_loss"](MWR_loss)
            self.valid_metrics["MUR_loss"](MUR_loss)

        return adversarial_loss
