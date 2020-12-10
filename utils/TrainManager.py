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
            self.train_metrics[name] = tf.keras.metrics.Mean()
            self.valid_metrics[name] = tf.keras.metrics.Mean()

    def init_training_hparams(self, BatchPerEpoch, epochs, T2=30):
        self.alpha = 1.0
        self.N = BatchPerEpoch
        self.T2 = T2
        self.d = self.alpha / (self.T2 * self.N)

    def compile(self, optimizer):
        self.optimizer = optimizer

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

            current_time = dt.now().strftime("%Y%m%d-%H%M%S")

            train_log_dir = os.path.join(tensorboard_log_dir, current_time, "train")
            valid_log_dir = os.path.join(tensorboard_log_dir, current_time, "test")

            self.train_summary_writter = tf.summary.create_file_writer(train_log_dir)
            self.valid_summary_writter = tf.summary.create_file_writer(valid_log_dir)

        for epoch in range(epochs):
            if verbose:
                print(f"{epoch + 1}/{epochs} epochs...")

            print("start train")
            for batch in train_dataloader:
                # self._train_batch(batch, self.alpha > 0, training=True)
                self.distributed_train_batch(batch, training=True)

            print("start validation")
            for batch in test_dataloader:
                # self._train_batch(batch, self.alpha > 0, training=False)
                self.distributed_train_batch(batch, training=False)

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

    def _write_on_tensorboard(self, epoch):
        with self.train_summary_writter.as_default():
            for key, value in self.train_metrics.items():
                tf.summary.scalar(key, value.result(), epoch)

        with self.valid_summary_writter.as_default():
            for key, value in self.valid_metrics.items():
                tf.summary.scalar(key, value.result(), epoch)

    @tf.function
    def distributed_train_batch(self, data, training):
        print("어디서 에러가 나는거야")
        per_replica_losses = self.strategy.experimental_run_v2(
            self._train_batch, args=(data, self.alpha > 0, training)
        )

        print("어디서 에러가 나는거야2")
        loss = self.strategy.reduce(
            tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None
        )

        print("어디서 에러가 나는거야3")
        return loss if training else per_replica_losses

    def _train_batch(
        self,
        data: Tuple,
        compute_auxiliary: bool = False,
        training: bool = True,
    ):
        print("one batch")
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

                WOR_pred = self.model(wor[0], task="WOR")
                # UOR_pred = self.model(uor[0], task="UOR")
                MWR_pred = self.model(mwr[0], task="MCR")
                MUR_pred = self.model(mwr[0], task="MCR")

                WOR_loss += self.MLE_loss(wor[1], WOR_pred)
                # UOR_loss += self.loss(uor[1], UOR_pred)
                MWR_loss += self.MCR_loss(mwr[1], MWR_pred)
                MUR_loss += self.MCR_loss(mwr[1], MUR_pred)

                # auxiliary_loss += WOR_loss + UOR_loss + MWR_loss + MUR_loss
                auxiliary_loss += WOR_loss + MWR_loss + MUR_loss

            adversarial_loss = mle_loss + self.alpha * auxiliary_loss

        if training:
            print(1)
            gradient = tape.gradient(adversarial_loss, self.model.trainable_variables)
            print(2)
            self.optimizer.apply_gradients(
                zip(gradient, self.model.trainable_variables)
            )
            print(3)

            self.train_metrics["adversarial_loss"].update_state(adversarial_loss)
            self.train_metrics["MLE_loss"].update_state(mle_loss)
            self.train_metrics["Auxiliary_loss"].update_state(auxiliary_loss)
            self.train_metrics["WOR_loss"].update_state(WOR_loss)
            # self.train_metrics["UOR_loss"].update_state(UOR_loss)
            self.train_metrics["MWR_loss"].update_state(MWR_loss)
            self.train_metrics["MUR_loss"].update_state(MUR_loss)
            print(4)

        else:
            self.valid_metrics["adversarial_loss"].update_state(adversarial_loss)
            self.valid_metrics["MLE_loss"].update_state(mle_loss)
            self.valid_metrics["Auxiliary_loss"].update_state(auxiliary_loss)
            self.valid_metrics["WOR_loss"].update_state(WOR_loss)
            # self.valid_metrics["UOR_loss"].update_state(UOR_loss)
            self.valid_metrics["MWR_loss"].update_state(MWR_loss)
            self.valid_metrics["MUR_loss"].update_state(MUR_loss)

        return adversarial_loss
