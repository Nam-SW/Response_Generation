from konlpy.tag import Okt
from soynlp.normalizer import repeat_normalize


okt = Okt()


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
