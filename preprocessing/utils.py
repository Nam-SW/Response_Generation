import os
import re

# from google.colab import drive
from konlpy.tag import Okt
from soynlp.normalizer import repeat_normalize


path = "data/raw_data"
talk_data_list = sorted([f for f in os.listdir(path) if f.endswith(".csv")])
utterance_size = 4

okt = Okt()
names = [
    "승우",
    "남승우",
    "세연",
    "이세연",
    "이세",
    "연진",
    "최연진",
    "정민",
    "최정민",
    "명준",
    "차명준",
    "준표",
    "이준표",
    "다빈",
    "최다빈",
    "희정",
    "유희정",
    "규리",
    "김규리",
    "큐리",
    "나연",
    "김나연",
    "배영",
    "공배영",
    "하림",
    "서하림",
    "화준",
    "세영",
    "오세영",
    "의현",
    "이의현",
    "익현",
    "우진",
    "손우진",
    "지웅",
    "김지웅",
    "찌찌웅",
    "찌웅",
    "지원",
    "정지원",
]
names = "|".join(names)


def filtering(text):
    def check_typo(text):
        if re.match("[ㄱ-ㅎ]+", text):
            text_set = set(text)
            if len({"ㅋ", "ㅌ", "ㄴ"} - text_set) < 2:
                text = "ㅋ" * len(text)

            elif {"ㅎ", "ㄹ"} == text_set:
                text = "ㅎ" * len(text)

        return text

    text = okt.normalize(text.lower())
    text = check_typo(text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(
        r"[^ .,?!%~\^\[\]\-_가-힣ㄱ-ㅎㅏ-ㅣa-z0-9]+|http.+|(?<=\d),(?=\d)", "", text
    )

    text = repeat_normalize(text, 5)
    # text = re.sub(r'(.+?)\1+', r'\1', text)
    text = re.sub(r"(.{3,}?)\1+", r"\1", text)
    return text
