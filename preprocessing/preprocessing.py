import os
from multiprocessing import Pool, cpu_count
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences

from preprocessing.utils import filtering, names, talk_data_list


global_tokenizer = None


def preprocessing(df):
    df = df.dropna()
    df = df.drop(df[df["contents"].str.startswith("삭제된 메시지입니다.")].index)
    df = df.drop(df[df["contents"].str.startswith("파일:")].index)

    # for s in '이모티콘\n사진\n동영상'.split('\n'):
    #     df = df[df['contents'] != s]

    df["writer"] = df["writer"].apply(lambda x: 0 if x == "남승우" else 1)
    df["contents_original"] = df["contents"]
    df["contents"] = df["contents"].str.replace(r"사진 \d+장", "사진")
    df["contents"] = df["contents"].str.replace(r"^이모티콘 (?=\w+)", "")
    df["contents"] = df["contents"].apply(lambda x: filtering(x))
    df["contents"] = df["contents"].str.replace(names, "[NAME]")
    df = df[df["contents"].str.len() >= 5]
    df.reset_index(inplace=True, drop=True)
    df["ids"] = df["contents"].apply(lambda x: global_tokenizer.encode(x).ids)

    return df


def save_data(params):
    assert len(params) == 4, "params must be 4."
    df, utterance_size, filename, load = params

    if load and (False if filename is None else os.path.isfile(filename)):
        print(filename + " load\n", end="")

        data = pd.read_csv(filename, encoding="utf-8")
        for column in data.columns[:-1]:
            data[column] = data[column].apply(
                lambda x: list(map(int, x[1:-1].split(", ")))
            )

    else:
        print(filename + " preprocessing\n", end="")
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
        # for i in tqdm(range(len(df))):
        for i in range(len(df)):
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

            temp_dict["content"] += row["ids"] + [3]  # [SEP] 추가
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
    load: bool,
    utterance_size: Optional[int] = None,
    tokenizer=None,
    use_multiprocessing=True,
):
    df_name_list = [f"data/preprocessed_data/{f}" for f in talk_data_list]

    if load:
        df_list = [None for filename in talk_data_list]

    else:
        assert tokenizer is not None, "for preprocessing, tokenizer must require."
        assert (
            utterance_size is not None
        ), "for preprocessing, utterance_size must require."

        df_list = [
            preprocessing(pd.read_csv(f"data/raw_data/{filename}", encoding="utf-8"))
            for filename in talk_data_list
        ]

    global_tokenizer = tokenizer

    if use_multiprocessing:
        with Pool(cpu_count()) as p:
            data = p.map(
                save_data,
                [
                    (df, utterance_size, fname, load)
                    for df, fname in zip(df_list, df_name_list)
                ],
            )
    else:
        data = [
            save_data(df, utterance_size, fname, load)
            for df, fname in zip(df_list, df_name_list)
        ]

    return data


def get_data(
    load: bool,
    utterance_size: int,
    max_len: int,
    tokenizer=None,
    use_multiprocessing=True,
):
    fname = "data/preprocessed_data/data.csv"

    def data_func(param):
        cpu_n, data = param
        new_data = pd.DataFrame(
            columns=[f"context_{i+1}" for i in range(utterance_size)]
            + ["response", "y"]
        )
        print(f"cpu_n: {cpu_n}")

        # for i in tqdm(range(len(data))):
        for i in range(len(data)):
            row = data.iloc[i]
            contexts = row[:4].to_list()
            response = row[4]
            for j in range(1, len(response)):
                line = contexts + [response[:j], response[j]]
                new_data = new_data.append(
                    pd.Series(line, index=new_data.columns), ignore_index=True
                )

        return new_data

    data_list = get_preprocessed_data(
        load, utterance_size, tokenizer, use_multiprocessing
    )

    data = pd.concat(data_list)
    data.reset_index(inplace=True, drop=True)

    if load:
        new_data = pd.read_csv(fname, encoding="utf-8")
        for column in new_data.columns[:-1]:
            new_data[column] = new_data[column].apply(
                lambda x: list(map(int, x[1:-1].split(", ")))
            )

    else:
        split_n = len(data) / cpu_count()

        if use_multiprocessing:
            with Pool(cpu_count()) as p:
                new_data = p.map(
                    data_func,
                    [
                        (i + 1, data.iloc[int(i * split_n) : int((i + 1) * split_n)])
                        for i in range(cpu_count())
                    ],
                )
            new_data = pd.concat(new_data)
            new_data.reset_index(inplace=True, drop=True)
        else:
            new_data = [
                data_func(i + 1, data.iloc[int(i * split_n) : int((i + 1) * split_n)])
                for i in range(cpu_count())
            ]

        new_data.to_csv(fname, encoding="utf-8", index=None)

    context = np.transpose(
        [
            pad_sequences(new_data[col].to_list(), max_len)
            for col in new_data.columns[:utterance_size]
        ],
        axes=[1, 0, 2],
    )
    response = pad_sequences(new_data["response"].to_list(), max_len)
    y = new_data["y"].to_numpy(dtype=np.int32)

    con_train, con_test, y_train, y_test = train_test_split(
        context, y, test_size=0.1, random_state=1
    )
    res_train, res_test, y_train, y_test = train_test_split(
        response, y, test_size=0.1, random_state=1
    )

    return (con_train, res_train, y_train), (con_test, res_test, y_test)
