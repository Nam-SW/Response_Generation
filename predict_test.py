import argparse
import warnings

import tensorflow as tf

from dataloader import get_dataloader, get_tf_data
from modeling.model import DialogWithAuxility
from tokenizer import load_tokenizer
from utils.JsonManager import JsonManager


warnings.filterwarnings(action="ignore")
tf.get_logger().setLevel("ERROR")
tf.autograph.set_verbosity(3)

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], "GPU")
    except RuntimeError as e:
        print(e)


if __name__ == "__main__":
    json_manager = JsonManager(__file__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--cnt", default=5)
    parser.add_argument("--gpu_num", default=0)
    args = parser.parse_args()
    cnt = int(args.cnt)
    gpu_num = int(args.gpu_num)

    gpus = tf.config.experimental.list_physical_devices("GPU")
    tf.config.experimental.set_visible_devices(gpus[gpu_num], "GPU")

    tokenizer = load_tokenizer("config/tokenizer.json")
    cls_token = tokenizer.token_to_id("[CLS]")
    sep_token = tokenizer.token_to_id("[SEP]")

    _, test_dataloader = get_dataloader("data/model_use_data", 0.9, True)
    dataloader = get_tf_data(test_dataloader.load_data, 1)

    model_hparams = json_manager.load("config/model_hparams.json")
    model = DialogWithAuxility(**model_hparams)
    checkpoint = tf.train.Checkpoint(
        optimizer=tf.keras.optimizers.Adagrad(), model=model
    )
    checkpoint.restore(tf.train.latest_checkpoint("model"))

    for i, data in enumerate(dataloader):
        print(i, cnt)
        if i >= cnt:
            break

        mle = data[0]
        sample_context = mle[0][0]  # 배치중 첫번째 샘플만
        sample_y = mle[1]  # 배치중 첫번째 샘플만

        for i, c in enumerate(sample_context[0]):
            print(f"context{i+1}: {tokenizer.decode(c)}")
        print(f"response: {tokenizer.decode(sample_y[0])}")

        text_list = [cls_token]
        last_predicted_word = None

        while last_predicted_word != sep_token and len(text_list) <= model.max_len:
            sample_response = tf.constant([text_list], dtype=tf.int32)
            last_predicted_word = int(
                tf.argmax(
                    model((sample_context, sample_response))[0],
                    axis=-1,
                )
            )
            text_list.append(last_predicted_word)

        print(f"predict:{tokenizer.decode(text_list)}")
