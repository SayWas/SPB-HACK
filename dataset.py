import pandas as pd
import itertools
import random

from tqdm import tqdm


# pd.options.display.max_colwidth = -1
# pd.options.display.min_rows = 100


def load_dataset(path):
    df = pd.read_csv(path, sep=",", header=0)
    return df[["target_building_id", "target_address"]]


def create_pair_data(text, labels, size):
    data_dict = {}
    for x in pd.DataFrame({"text": text, "label": labels}).groupby("label").apply(
            lambda x: list(x["text"])).reset_index().values.tolist():
        data_dict[x[0]] = x[1]
    label1 = []
    for d in data_dict.keys():
        label1.extend(list(itertools.combinations(data_dict[d], 2)))

    label0 = []
    for d1 in tqdm(data_dict.keys()):
        for d2 in data_dict.keys():
            if len(label0) >= len(label1):
                break
            if d1 != d2:
                label0.extend(list(itertools.product(data_dict[d1], data_dict[d2])))
    label0 = random.sample(label0, len(label1))
    return random.sample(label1, int(size / 2)), random.sample(label0, int(size / 2))


def get_corpus(path):
    df = pd.read_csv(path, sep=",", header=0)
    df.drop_duplicates(subset=["full_address"], inplace=True)
    return df["full_address"].tolist()
