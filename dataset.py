from typing import Dict, List

import bz2
from itertools import tee

import numpy as np
import pandas as pd
import regex as re
import torch
from datasets import load_dataset
from rich.console import Console
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from utils import murmurhash

REGEX0 = re.compile(r"([\p{S}\p{P}]+)")
REGEX1 = re.compile(r" +")

console = Console()


class DummyDataset(Dataset):
    def __init__(
        self,
        x,
        has_label: bool = True,
        feature_size: int = 512,
        add_eos_tag: bool = True,
        add_bos_tag: bool = True,
        max_seq_len: int = 256,
        label2index: Dict[str, int] = None,
    ):
        self.feature_size = feature_size
        self.add_eos_tag = add_eos_tag
        self.add_bos_tag = add_bos_tag
        self.max_seq_len = max_seq_len
        self.label2index = label2index
        self.has_label = has_label

        self.x = x
        self.eos_tag_ = 1 if add_eos_tag else 0
        self.bos_tag_ = 1 if add_bos_tag else 0
        self.eos_tag = ["EOS"] if add_eos_tag else []
        self.bos_tag = ["BOS"] if add_bos_tag else []

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        line = self.x[idx]
        text = line["text"]
        if self.has_label:
            label = line["label"]
            if self.label2index:
                label = self.label2index[label]
        else:
            label = None

        tokens = (
            REGEX1.sub(" ", REGEX0.sub(r" \1 ", text.lower()))
            .strip()
            .split(" ")
        )

        curr_tokens = self.bos_tag + tokens[: self.max_seq_len] + self.eos_tag
        curr_hashings = []
        for j in range(len(curr_tokens)):
            curr_hashing = murmurhash(
                curr_tokens[j], feature_size=self.feature_size
            )
            curr_hashings.append(curr_hashing[: self.feature_size // 2])
        if label is not None:
            return curr_hashings, label
        else:
            return curr_hashings


def collate_fn(examples):
    projection = []
    labels = []

    for example in examples:
        if not isinstance(example, tuple):
            projection.append(np.asarray(example))
        else:
            projection.append(np.asarray(example[0]))
            labels.append(example[1])
    lengths = torch.from_numpy(np.asarray(list(map(len, examples)))).long()
    projection_tensor = np.zeros(
        (len(projection), max(map(len, projection)), len(projection[0][0]))
    )
    for i, doc in enumerate(projection):
        projection_tensor[i, : len(doc), :] = doc

    return (
        torch.from_numpy(projection_tensor).float(),
        lengths,
        torch.from_numpy(np.asarray(labels)),
    )


def create_dataloaders(
    task: str = "yelp",
    batch_size: int = 32,
    feature_size: int = 128,
    add_eos_tag: bool = True,
    add_bos_tag: bool = True,
    max_seq_len: int = 256,
    label2index: Dict[str, int] = None,
):
    if task == "yelp":
        dataset = load_dataset("yelp_polarity")
    elif task == "yelp-5":
        data = pd.read_json("data/yelp_reviews.json", lines=True)
        data["label"] = data["stars"] - 1
        train, val = train_test_split(
            data, test_size=0.1, stratify=data["label"]
        )
        dataset = {
            "train": train.to_dict("records"),
            "test": val.to_dict("records"),
        }
    else:
        raise Exception(f"Unsupported task: {task} VS. yelp")

    train_dataset = DummyDataset(
        dataset["train"],
        feature_size=feature_size,
        add_eos_tag=add_eos_tag,
        add_bos_tag=add_bos_tag,
        max_seq_len=max_seq_len,
        label2index=label2index,
    )

    val_dataset = DummyDataset(
        dataset["test"],
        feature_size=feature_size,
        add_eos_tag=add_eos_tag,
        add_bos_tag=add_bos_tag,
        max_seq_len=max_seq_len,
        label2index=label2index,
    )

    return (
        DataLoader(
            train_dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=4,
        ),
        DataLoader(
            val_dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=4,
        ),
    )
