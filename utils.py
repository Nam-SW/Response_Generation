import json
import os


class JsonManager:
    def __init__(self, basepath: str):
        basepath = os.path.dirname(basepath)
        self.basepath = os.path.abspath(basepath)

    def load(self, path):
        with open(os.path.join(self.basepath, path), mode="r", encoding="utf-8") as f:
            return json.load(f)
