from typing import Dict, Tuple, List

import tensorflow as tf

# from modeling.model import DialogWithAuxility


class TrainManager:
    def __init__(self, model, **kwargs):
        self.model = model

    def train(self, train_data, validation_data=None):
        pass
