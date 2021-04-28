import json
import os
import re

from soynlp import repeat_normalize


class JsonManager:
    def __init__(self, basepath: str):
        basepath = os.path.dirname(basepath)
        self.basepath = os.path.abspath(basepath)

    def load(self, filename):
        with open(
            os.path.join(self.basepath, filename), mode="r", encoding="utf-8"
        ) as f:
            data = json.load(f)
        return data

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


def discord_filtering():
    pass


def filtering(text):
    def check_typo(text):
        if re.match("[ㄱ-ㅎ]+", text):
            text_set = set(text)
            if len({"ㅋ", "ㅌ", "ㄴ"} - text_set) < 2:
                text = "ㅋ" * len(text)

            elif {"ㅎ", "ㄹ"} == text_set:
                text = "ㅎ" * len(text)

        return text

    text = check_typo(text.strip().lower())
    text = re.sub(r"사진 \d+장", "사진", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(
        r"[^ .,?!%~\^\[\]\-_가-힣ㄱ-ㅎㅏ-ㅣa-z0-9]+|http.+|(?<=\d),(?=\d)", "", text
    )

    text = repeat_normalize(text, 6)
    text = re.sub(r"(.{6,}?)\1+", r"\1", text)
    return text
