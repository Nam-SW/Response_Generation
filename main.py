import argparse

import utils

from preprocessing.preprocessing import get_data
from preprocessing.tokenizer import get_tokenizer
from modeling.training import model_training, model_evaluate


json_manager = utils.JsonManager(__file__)

parser = argparse.ArgumentParser(
    description="build and training a DialogWithAuxility Model"
)
parser.add_argument("--data_hparams", default="{}")
parser.add_argument("--model_hparams", default="{}")
parser.add_argument("--load_data", default=True)
parser.add_argument("--load_model", default=True)
args = parser.parse_args()

data_hparams = json_manager.load(args.data_hparams)
model_hparams = json_manager.load(args.model_hparams)
load_data = (args.load_data.lower()) == "true"
load_model = (args.load_model.lower()) == "true"


data_hparams["tokenizer"] = get_tokenizer(
    data_hparams["limit_alphabet"],
    data_hparams["vocab_size"],
    load_data,
    data_hparams.get("additional_special_tokens", None),
)

train_data, test_data = get_data(
    load_data,
    data_hparams["utterance_size"],
    data_hparams["max_len"],
    data_hparams["tokenizer"],
    data_hparams["use_multiprocessing"],
)

model = model_training(train_data, model_hparams, load=load_model)
model_evaluate(test_data, model)

model.save_weight("model_weight.h5")
