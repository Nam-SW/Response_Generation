import os
from multiprocessing import Pool, cpu_count
from typing import List, Tuple

import pandas as pd
from tokenizer import load_tokenizer
from tqdm import tqdm
from utils.filtering import filtering


def preprocessing(df: pd.DataFrame, remove_names: List[str], tokenizer):
    df = df.dropna()
    df = df.drop(df[df["contents"].str.startswith("삭제된 메시지입니다.")].index)
    df = df.drop(df[df["contents"].str.startswith("파일:")].index)

    writer_dict = dict(map(lambda x: x[::-1], enumerate(df["writer"].unique())))

    df["writer"] = df["writer"].apply(lambda x: writer_dict[x])
    df["contents_original"] = df["contents"]
    df["contents"] = df["contents"].apply(lambda x: x.strip() + " ")
    df["contents"] = df["contents"].str.replace(r"사진 \d+장", "사진")
    df["contents"] = df["contents"].str.replace(r"^이모티콘 (?=\w+)", "")
    df["contents"] = df["contents"].apply(filtering)
    df["contents"] = df["contents"].str.replace(remove_names, "[NAME]")
    # df = df[df["contents"].str.len() >= 2]
    df.reset_index(inplace=True, drop=True)
    df["ids"] = list(
        map(lambda x: x.ids, tokenizer.encode_batch(df["contents"].to_list()))
    )

    return df


def save_data(params: Tuple[pd.DataFrame, int, str]):
    assert len(params) == 3, "params must be 3."
    df, utterance_size, filename = params

    if filename and os.path.isfile(filename):
        df = pd.read_csv(filename, encoding="utf-8")
        for column in df.columns[:-1]:
            df[column] = df[column].apply(lambda s: list(map(int, s[1:-1].split(", "))))
        return df

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

        temp_dict["content"] += row["ids"]
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
    if use_multiprocessing:
        with Pool(cpu_count()) as p:
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

    for i in tqdm(range(len(data))):
        row = data.iloc[i]
        contexts = row[:4].to_list()
        response = row[4][:-1]
        y = row[4][1:]
        if len(y) < 3:
            continue
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
    if os.path.isfile(fname):
        return pd.read_csv(fname, encoding="utf-8")

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

    return new_data


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
    if not os.path.isdir(preprocessed_data_dir):
        os.mkdir(preprocessed_data_dir)
    preprocessed_data_dir = os.path.abspath(preprocessed_data_dir)
    if not os.path.isdir(model_use_data_dir):
        os.mkdir(model_use_data_dir)
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
    data = get_data(
        data_list,
        utterance_size,
        max_len,
        model_use_data_dir,
        use_multiprocessing,
    )
    data.to_csv(
        os.path.join(model_use_data_dir, "data.csv"),
        encoding="utf-8",
        index=None,
    )
    # data.sample(frac=1).reset_index(drop=True)

    # i = int(len(data) * train_test_split)
    # train_data = data.iloc[:-i]
    # valid_data = data.iloc[-i:]
    # train_data.to_csv(
    #     os.path.join(model_use_data_dir, "train_data.csv"),
    #     encoding="utf-8",
    #     index=None,
    # )
    # valid_data.to_csv(
    #     os.path.join(model_use_data_dir, "valid_data.csv"),
    #     encoding="utf-8",
    #     index=None,
    # )
