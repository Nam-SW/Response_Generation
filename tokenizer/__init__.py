import os
from typing import List

import pandas as pd
from tokenizers import BertWordPieceTokenizer, Tokenizer

from utils import filtering


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


def setup_tokenizer_train_data(data_dir: str, remove_names: List[str]):
    data_dir = os.path.abspath(data_dir)
    remove_names = "|".join(remove_names)

    df_list = [
        pd.read_csv(os.path.join(data_dir, filename), encoding="utf-8")
        for filename in os.listdir(data_dir)
        if filename.endswith(".csv")
    ]
    all_df = pd.concat(df_list)

    all_df.dropna(inplace=True)
    all_df.reset_index(inplace=True, drop=True)
    all_df["contents"] = all_df["contents"].apply(lambda x: filtering(x))
    all_df["contents"] = all_df["contents"].str.replace(remove_names, "[NAME]")
    all_df = all_df[
        (all_df["contents"].str.len() < 70) & (all_df["contents"].str.len() >= 5)
    ]

    filename = os.path.join(data_dir, "tokenizer_train_data.txt")
    with open(filename, "w", encoding="utf-8") as f:
        for line in all_df["contents"]:
            f.write(line + "\n")


def train_tokenizer(
    limit_alphabet: int,
    vocab_size: int,
    remove_names: List[str],
    data_dir: str,
    save_dir: str = "config",
):
    save_dir = os.path.abspath(save_dir)
    data_dir = os.path.abspath(data_dir)

    setup_tokenizer_train_data(data_dir, remove_names)

    tokenizer = BertWordPieceTokenizer(
        clean_text=True,
        handle_chinese_chars=True,
        strip_accents=False,  # Must be False if cased model
        lowercase=False,
        wordpieces_prefix="##",
    )

    files = [
        os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".txt")
    ]
    tokenizer.train(
        files=files,
        limit_alphabet=limit_alphabet,
        vocab_size=limit_alphabet,
        special_tokens=special_tokens,
    )
    filename = os.path.join(save_dir, "tokenizer.json")
    tokenizer.save(filename)
    print("tokenizer train and save at " + filename)


def load_tokenizer(filename):
    filename = os.path.abspath(filename)
    return Tokenizer.from_file(filename)
