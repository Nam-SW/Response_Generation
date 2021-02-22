import os
from copy import deepcopy
from datetime import datetime as dt
from re import search
from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tokenizer import load_tokenizer


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=5000):
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


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
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction=tf.keras.losses.Reduction.NONE,
        )

        def mle_loss(y_true, y_pred):
            loss = loss_object(y_true, y_pred)
            mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
            loss = tf.multiply(loss, mask)
            # return tf.nn.compute(
            #     loss, global_batch_size=self.global_batch_size
            # )
            return tf.reduce_mean(loss)

        def mcr_loss(x, y_true, y_pred):
            loss = loss_object(y_true, y_pred)
            mask = tf.cast(tf.not_equal(y_true, 4), tf.float32)
            loss = tf.multiply(loss, mask)
            # return tf.nn.compute_average_loss(
            #     loss, global_batch_size=self.global_batch_size
            # )
            return tf.reduce_mean(loss)

        self.MLE_loss = mle_loss
        self.MCR_loss = mcr_loss

    def init_metrics(self):
        self.train_metrics = {}
        self.valid_metrics = {}

        for name in self.metrics_name:
            self.train_metrics[name] = tf.keras.metrics.Mean("train_" + name)
            self.valid_metrics[name] = tf.keras.metrics.Mean("valid_" + name)

    def init_training_hparams(self, BatchPerEpoch, T1, T2=30):
        self.alpha = 1.0
        self.N = BatchPerEpoch
        self.T1 = T1
        self.T2 = T2
        self.d = self.alpha / (self.T2 * self.N)
        self.train_global_step = 0

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
        tensorboard_log_dir: Optional[str] = None,
        global_max_step: int = 50000,
        validation_step: int = 1000,
        verbose: int = 1,
        test_tokenizer_config: Optional[str] = None,
        load_latest: bool = False,
    ):
        assert self.optimizer is not None, "model must be compiled before training."

        self.strategy = strategy
        self.init_training_hparams(BatchPerEpoch, global_max_step)

        if not os.path.isdir(model_save_dir):
            os.mkdir(model_save_dir)
        model_save_dir = os.path.abspath(model_save_dir)
        model_save_prefix = os.path.join(model_save_dir, "ckpt")

        checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)

        if load_latest:
            try:
                checkpoint.restore(tf.train.latest_checkpoint(model_save_dir))
                fn = sorted(os.listdir(model_save_dir))[-1]
                lasted_step = int(search(r"\d+", fn).group()) * validation_step
                self.alpha = max(0, self.alpha - self.d * lasted_step)
                self.train_global_step = lasted_step

                print("Checkpoint loaded successfully.")
                print(f"Starts from the last saved step {lasted_step}.")
            except Exception:
                print("Can not load latest checkpoint.")

        if tensorboard_log_dir:
            tensorboard_log_dir = os.path.abspath(tensorboard_log_dir)
            self.setup_tensorboard(tensorboard_log_dir)
            self.use_tensorboard = True

        # TODO: 테스트용 텍스트 생성 구현하기
        test_predict = test_tokenizer_config is not None
        if test_predict:
            test_tokenizer_config = os.path.abspath(test_tokenizer_config)
            tokenizer = load_tokenizer(test_tokenizer_config)

            cls_token = tokenizer.token_to_id("[CLS]")
            sep_token = tokenizer.token_to_id("[SEP]")

            for batch in test_dataloader:
                sample_data = batch["MLE"]  # 배치중 첫번째 샘플만
                sample_data["context_ids"] = sample_data["context_ids"][:1]
                sample_y = sample_data["y"][:1]  # 배치중 첫번째 샘플만

                with open("predict_test.txt", "w", encoding="utf-8") as f:
                    f.write(
                        f"context: {tokenizer.decode(sample_data['context_ids'][0])}\n"
                    )
                    f.write(f"\nresponse: {tokenizer.decode(sample_y[0])}\n\n\n")

                break

        print(
            "Train Start.\n"
            + f"Learn {global_max_step} steps, and save the model every {validation_step} step."
            + f"and BatchPerEpoch is {BatchPerEpoch}."
        )
        for batch in train_dataloader:
            # 학습
            self.distributed_train_batch(batch, self.alpha > 0, training=True)
            self._write_on_tensorboard_train()

            # validation_step 개의 배치를 돌았으면
            if (self.train_global_step + 1) % validation_step == 0:
                # validation
                for batch in test_dataloader:
                    self.distributed_train_batch(batch, self.alpha > 0, training=False)

                # 출력
                if verbose:
                    print(f"{self.train_global_step + 1} Step")
                    for value in self.valid_metrics.values():
                        print(
                            f"valid_{value.name}: {value.result(): .4f}",
                            end="\t",
                        )
                    print("\n")

                # 텐서보드 작성. 1000에포크 단위
                self._write_on_tensorboard_valid(
                    self.train_global_step // validation_step
                )
                # 모델 저장
                checkpoint.save(model_save_prefix)

                if test_predict:
                    text_list = [cls_token]
                    last_predicted_word = None

                    while (
                        last_predicted_word != sep_token
                        and len(text_list) <= self.model.max_len
                    ):
                        sample_data["response_ids"] = tf.constant(
                            [text_list], dtype=tf.int32
                        )
                        pred = self.model(sample_data)[0]
                        last_predicted_word = int(
                            tf.argmax(
                                pred[-1],
                                axis=-1,
                            )
                        )
                        text_list.append(last_predicted_word)

                    with open("predict_test.txt", "a+", encoding="utf-8") as f:
                        f.write(
                            "predict_step_{}: {}\n".format(
                                self.train_global_step + 1, str(text_list)
                            )
                        )
                        f.write(
                            "predict_step_{}: {}\n\n".format(
                                self.train_global_step + 1,
                                tokenizer.decode(text_list),
                            )
                        )

            # 1회 배치가 끝나고, 파라미터 조정
            self.alpha = max(0, self.alpha - self.d)
            self.train_global_step += 1

            # 학습 끝
            if self.train_global_step > self.T1:
                break

        print(f"train Done! model saved at {model_save_dir}")

    def _write_on_tensorboard_train(self):
        if not self.use_tensorboard:
            return
        with self.train_summary_writter.as_default():
            for value in self.train_metrics.values():
                tf.summary.scalar(value.name, value.result(), self.train_global_step)
                value.reset_states()

    def _write_on_tensorboard_valid(self, step):
        if not self.use_tensorboard:
            return
        with self.valid_summary_writter.as_default():
            for value in self.valid_metrics.values():
                tf.summary.scalar(value.name, value.result(), step)
                value.reset_states()

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
        with tf.GradientTape() as tape:
            mle_pred = self.model(
                data["MLE"],
                task="MLE",
                training=training,
            )
            mle_loss = self.MLE_loss(data["MLE"]["y"], mle_pred)
            auxiliary_loss = 0
            # loss_dict = {task: 0 for task in ["WOR", "UOR", "MWR", "MUR"]}
            loss_dict = {task: 0 for task in ["WOR", "MWR", "MUR"]}

            # if compute_auxiliary:
            #     for task in loss_dict.keys():
            #         task_data = data[task]
            #         pred = self.model(task_data, task=task, training=training)
            #         if task == "WOR":
            #             loss = self.MLE_loss(task_data["y"], pred)
            #         elif task in ["MWR", "MUR"]:
            #             loss = self.MCR_loss(
            #                 task_data["context_ids"], task_data["y"], pred
            #             )
            #         elif task == "UOR":
            #             loss = 0  # TODO: implement
            #         loss_dict[task] = loss

            #     auxiliary_loss += sum(loss_dict.values())

            adversarial_loss = mle_loss + self.alpha * auxiliary_loss

        if training:
            gradient = tape.gradient(adversarial_loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(
                zip(gradient, self.model.trainable_variables)
            )

            self.train_metrics["adversarial_loss"](adversarial_loss)
            self.train_metrics["MLE_loss"](mle_loss)
            # self.train_metrics["Auxiliary_loss"](auxiliary_loss)
            # for task, loss in loss_dict.items():
            #     self.train_metrics[f"{task}_loss"](loss)

        else:
            self.valid_metrics["adversarial_loss"](adversarial_loss)
            self.valid_metrics["MLE_loss"](mle_loss)
            # self.valid_metrics["Auxiliary_loss"](auxiliary_loss)
            # for task, loss in loss_dict.items():
            #     self.valid_metrics[f"{task}_loss"](loss)

        return adversarial_loss
