import json
import os
import re


class JsonManager:
    def __init__(self, basepath: str):
        basepath = os.path.dirname(basepath)
        self.basepath = os.path.abspath(basepath)

    def load(self, filename):
        with open(
            os.path.join(self.basepath, filename), mode="r", encoding="utf-8"
        ) as f:
            return json.load(f)

    def save(self, filename, file):
        with open(
            os.path.join(self.basepath, filename), mode="w", encoding="utf-8"
        ) as f:
            json.dump(file, f)
