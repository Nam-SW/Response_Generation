import warnings
from math import ceil

import tensorflow as tf

from dataloader import get_dataloader, get_tf_data
from modeling.model import DialogWithAuxility
from utils.TrainManager import TrainManager


warnings.filterwarnings(action="ignore")
tf.get_logger().setLevel("ERROR")
tf.autograph.set_verbosity(3)


def train(args, model_hparams):
    # 파라미터
    gpus = tf.config.experimental.list_physical_devices("GPU")
    gpu_count = max(min(args.gpu_count, len(gpus)), 1)
    tf.config.experimental.set_visible_devices(gpus[:gpu_count], "GPU")
    gpus = [gpu.name.split(":", 1)[-1] for gpu in gpus]
    batch_size = args.batch_size
    strategy = tf.distribute.MirroredStrategy(gpus[:gpu_count])

    # 모델, 데이터셋 준비
    with strategy.scope():
        train_dataloader, test_dataloader = get_dataloader(
            args.data_dir,
            args.validation_split,
            args.data_shuffle,
        )

        train_dist_dataset = strategy.experimental_distribute_dataset(
            get_tf_data(train_dataloader.load_data, gpu_count * batch_size)
        )
        test_dist_dataset = strategy.experimental_distribute_dataset(
            get_tf_data(test_dataloader.load_data, gpu_count * batch_size, repeat=False)
        )

        model = DialogWithAuxility(**model_hparams)

        # 학습
        trainer = TrainManager(model, gpu_count, batch_size)
        trainer.compile(tf.keras.optimizers.Adagrad(args.learning_rate))
        trainer.train(
            train_dist_dataset,
            test_dist_dataset,
            strategy=strategy,
            BatchPerEpoch=ceil(len(train_dataloader) / (batch_size * gpu_count)),
            model_save_dir=args.model_save_dir,
            tensorboard_log_dir=args.tensorboard_log_dir,
            global_max_step=args.global_max_step,
            validation_step=args.validation_step,
            verbose=args.verbose,
            test_tokenizer_config=args.tokenizer,
            load_latest=args.load_latest,
        )
