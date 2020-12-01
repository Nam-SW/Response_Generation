import os
from typing import Optional, List

import pandas as pd
from tokenizers import BertWordPieceTokenizer

from preprocessing.utils import *


# limit_alphabet = int(input("limit_alphabet: "))
# vocab_size = int(input("vocab_size: "))
# limit_alphabet = 6000
# vocab_size = 8000
# load = input("file load? (y or n): ").lower() == "y"

special_tokens = [
    "[PAD]",
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "[MASK]",
    "[SEPT]",
    "[NAME]",
    "[BOS]",
    "[EOS]",
    "[UNK0]",
    "[UNK1]",
    "[UNK2]",
    "[UNK3]",
    "[UNK4]",
    "[UNK5]",
    "[UNK6]",
    "[UNK7]",
    "[UNK8]",
    "[UNK9]",
] + [f"[unused{i}]" for i in range(100)]


def get_tokenizer(
    limit_alphabet: int,
    vocab_size: int,
    load: bool = True,
    additional_special_tokens: Optional[List[str]] = None,
):
    filepath = f"tokenizer/vocab_{limit_alphabet}_{vocab_size}-vocab.txt"

    if os.path.isfile(filepath) and load:
        tokenizer = BertWordPieceTokenizer(vocab_file=filepath)
        df_list = [None for filename in talk_data_list]

    else:
        df_list = [
            pd.read_csv(f"{path}/{filename}", encoding="utf-8")
            for filename in talk_data_list
        ]
        all_df = pd.concat(df_list)

        all_df.dropna(inplace=True)
        all_df.reset_index(inplace=True, drop=True)
        all_df["contents"] = all_df["contents"].apply(lambda x: filtering(x))
        all_df["contents"] = all_df["contents"].str.replace(names, "[NAME]")
        # all_df = all_df[all_df['contents'].str.len() >= 5]
        all_df = all_df[
            (all_df["contents"].str.len() < 70) & (all_df["contents"].str.len() >= 5)
        ]

        if additional_special_tokens:
            special_tokens += additional_special_tokens

        with open("tokenizer/tokenizer_train_data.txt", "w", encoding="utf-8") as f:
            for line in all_df["contents"]:
                f.write(line + "\n")

        tokenizer = BertWordPieceTokenizer(
            clean_text=True,
            handle_chinese_chars=True,
            strip_accents=False,  # Must be False if cased model
            lowercase=False,
            wordpieces_prefix="##",
        )

        print("train")
        tokenizer.train(
            files=["tokenizer/tokenizer_train_data.txt"],
            limit_alphabet=limit_alphabet,
            vocab_size=vocab_size,
            special_tokens=special_tokens,
        )

        tokenizer.save("tokenizer", f"vocab_{limit_alphabet}_{vocab_size}")

    return tokenizer
