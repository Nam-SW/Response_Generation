import json
import os


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


def dump_jsonl(data, output_path, append=False):
    mode = "a+" if append else "w"
    with open(output_path, mode, encoding="utf-8") as f:
        for line in data:
            json_record = json.dumps(line, ensure_ascii=False)
            f.write(json_record + "\n")


def load_jsonl(input_path) -> list:
    data = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.rstrip("\n|\r")))
    return data
