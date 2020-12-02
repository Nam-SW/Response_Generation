import os
from multiprocessing import Pool, cpu_count
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm

from dataloader.utils import filtering
from tokenizer import load_tokenizer


def preprocessing(df: pd.DataFrame, remove_names: List[str], tokenizer):
    df = df.dropna()
    df = df.drop(df[df["contents"].str.startswith("삭제된 메시지입니다.")].index)
    df = df.drop(df[df["contents"].str.startswith("파일:")].index)

    writer_dict = dict(map(lambda x: x[::-1], enumerate(df["writer"].unique())))

    df["writer"] = df["writer"].apply(lambda x: writer_dict[x])
    df["contents_original"] = df["contents"]
    df["contents"] = df["contents"].str.replace(r"사진 \d+장", "사진")
    df["contents"] = df["contents"].str.replace(r"^이모티콘 (?=\w+)", "")
    df["contents"] = df["contents"].apply(lambda x: filtering(x))
    df["contents"] = df["contents"].str.replace(remove_names, "[NAME]")
    df = df[df["contents"].str.len() >= 5]
    df.reset_index(inplace=True, drop=True)
    df["ids"] = df["contents"].apply(lambda x: tokenizer.encode(x).ids)

    return df


def save_data(params: Tuple[pd.DataFrame, int, str]):
    assert len(params) == 3, "params must be 3."
    df, utterance_size, filename = params

    data = pd.DataFrame(
        columns=[f"utterance_{i+1}" for i in range(utterance_size)]
        + ["label", "original"]
    )

    initial_content = [2]
    temp_dict = {
        "content": initial_content.copy(),
        "last_original": "",
        "writer": None,
        "utterances": [initial_content.copy() for _ in range(utterance_size + 1)],
    }
    for i in tqdm(range(len(df))):
        row = df.iloc[i]
        if (
            temp_dict["writer"] != row["writer"]
            and temp_dict["content"] != initial_content
        ):  # 발화자가 다를때 비어있지 않으면
            temp_dict["content"].append(3)  # [SEP] 추가
            temp_dict["utterances"].append(temp_dict["content"])
            temp_dict["utterances"] = temp_dict["utterances"][1:]

            line = temp_dict["utterances"] + [temp_dict["last_original"]]
            data = data.append(
                pd.Series(line, index=data.columns), ignore_index=True
            )  # 추가

            temp_dict["content"] = initial_content.copy()
            temp_dict["last_original"] = ""

        temp_dict["last_original"] += row["contents_original"]
        temp_dict["writer"] = row["writer"]

    temp_dict["utterances"].append(temp_dict["content"])
    temp_dict["utterances"] = temp_dict["utterances"][1:]
    line = temp_dict["utterances"] + [temp_dict["last_original"]]
    data = data.append(pd.Series(line, index=data.columns), ignore_index=True)  # 추가

    if filename:
        data.to_csv(filename, encoding="utf-8", index=None)

    return data


def get_preprocessed_data(
    raw_data_dir: str,
    preprocessed_data_dir: str,
    remove_names: str,
    utterance_size: int,
    tokenizer,
    use_multiprocessing=True,
):
    raw_df_names = [
        os.path.join(raw_data_dir, f)
        for f in os.listdir(raw_data_dir)
        if f.endswith(".csv")
    ]
    df_list = [
        preprocessing(pd.read_csv(f), remove_names, tokenizer) for f in raw_df_names
    ]

    preprocess_df_names = [
        os.path.join(preprocessed_data_dir, f)
        for f in os.listdir(raw_data_dir)
        if f.endswith(".csv")
    ]
    # TODO: with 구문에서 빠져나오질 못함
    if use_multiprocessing:
        with Pool(cpu_count() // 2) as p:
            data = p.map(
                save_data,
                [
                    (df, utterance_size, fname)
                    for df, fname in zip(df_list, preprocess_df_names)
                ],
            )
    else:
        data = [
            save_data((df, utterance_size, fname))
            for df, fname in zip(df_list, preprocess_df_names)
        ]

    return data


def _data_func(param):
    data, utterance_size = param
    new_data = pd.DataFrame(
        columns=[f"context_{i+1}" for i in range(utterance_size)] + ["response", "y"]
    )

    for i in range(len(data)):
        row = data.iloc[i]
        contexts = row[:4].to_list()
        response = row[4][:-1]
        y = row[4][1:]
        line = contexts + [response, y]
        new_data = new_data.append(
            pd.Series(line, index=new_data.columns), ignore_index=True
        )

    return new_data


def get_data(
    data_list: List[pd.DataFrame],
    utterance_size: int,
    max_len: int,
    model_use_data_dir: str,
    use_multiprocessing=True,
):
    fname = os.path.join(model_use_data_dir, "data.csv")

    data = pd.concat(data_list)
    data.reset_index(inplace=True, drop=True)

    split_n = len(data) / cpu_count()

    if use_multiprocessing:
        with Pool(cpu_count()) as p:
            new_data = p.map(
                _data_func,
                [
                    (
                        data.iloc[int(i * split_n) : int((i + 1) * split_n)],
                        utterance_size,
                    )
                    for i in range(cpu_count())
                ],
            )

        new_data = pd.concat(new_data)
        new_data.reset_index(inplace=True, drop=True)
    else:
        new_data = _data_func((data, utterance_size))

    new_data.to_csv(fname, encoding="utf-8", index=None)

    context = np.transpose(
        [
            pad_sequences(
                new_data[col].to_list(), max_len, truncating="post", padding="post"
            )
            for col in new_data.columns[:utterance_size]
        ],
        axes=[1, 0, 2],
    )
    response = pad_sequences(
        new_data["response"].to_list(), max_len, truncating="post", padding="post"
    )
    y = pad_sequences(
        new_data["y"].to_list(), max_len, truncating="post", padding="post"
    )

    return context, response, y


def _save_whole_data(model_use_data_dir: str, context, response, y):
    context_fname = os.path.join(model_use_data_dir, "context.npy")
    response_fname = os.path.join(model_use_data_dir, "response.npy")
    y_fname = os.path.join(model_use_data_dir, "y.npy")

    np.save(context_fname, context)
    np.save(response_fname, response)
    np.save(y_fname, y)


def setup_data(
    raw_data_dir: str,
    preprocessed_data_dir: str,
    model_use_data_dir: str,
    remove_names: List[str],
    tokenizer_config: str,
    utterance_size: int,
    max_len: int,
    use_multiprocessing: bool = True,
):
    raw_data_dir = os.path.abspath(raw_data_dir)
    preprocessed_data_dir = os.path.abspath(preprocessed_data_dir)
    model_use_data_dir = os.path.abspath(model_use_data_dir)
    remove_names = "|".join(remove_names)
    tokenizer_config = os.path.abspath(tokenizer_config)

    tokenizer = load_tokenizer(tokenizer_config)

    print("preprocessing")
    data_list = get_preprocessed_data(
        raw_data_dir,
        preprocessed_data_dir,
        remove_names,
        utterance_size,
        tokenizer,
        use_multiprocessing,
    )

    print("data setting")
    context, response, y = get_data(
        data_list,
        utterance_size,
        max_len,
        model_use_data_dir,
        use_multiprocessing,
    )

    _save_whole_data(model_use_data_dir, context, response, y)


def load_data(data_dir: str, validation_split: float):
    data_dir = os.path.abspath(data_dir)

    context_fname = os.path.join(data_dir, "context.npy")
    response_fname = os.path.join(data_dir, "response.npy")
    y_fname = os.path.join(data_dir, "y.npy")

    csv_fname = os.path.join(data_dir, "data.csv")

    if (
        os.path.isfile(context_fname)
        and os.path.isfile(response_fname)
        and os.path.isfile(y_fname)
    ):
        context = np.load(context_fname)
        response = np.load(response_fname)
        y = np.load(y_fname)

    elif os.path.isfile(csv_fname):
        new_data = pd.read_csv(csv_fname, encoding="utf-8")
        for column in new_data.columns:
            new_data[column] = new_data[column].apply(
                lambda x: list(map(int, x[1:-1].split(", ")))
            )

    else:
        raise Exception("Data file Not found.")

    con_train, con_test, y_train, y_test = train_test_split(
        context, y, test_size=0.1, random_state=1
    )
    res_train, res_test, y_train, y_test = train_test_split(
        response, y, test_size=0.1, random_state=1
    )
    return (con_train, res_train, y_train), (con_test, res_test, y_test)
