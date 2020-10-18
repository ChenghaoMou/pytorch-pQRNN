from typing import Dict, List

import bz2
from itertools import tee

import numpy as np
import pandas as pd
import regex as re
import torch
from rich.console import Console
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from utils import murmurhash

REGEX0 = re.compile(r"([\p{S}\p{P}]+)")
REGEX1 = re.compile(r" +")

console = Console()


class DummyDataset(IterableDataset):
    def __init__(
        self,
        x,
        has_label: bool = True,
        feature_size: int = 128,
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

    def __iter__(self):
        corpus = bz2.BZ2File(self.x)
        for line in corpus:
            try:
                line = line.decode("utf-8")
                if self.has_label and self.label2index:
                    label, line = line.split(" ", 1)
                    label = self.label2index[label]
                else:
                    label = None

                tokens = (
                    REGEX1.sub(" ", REGEX0.sub(r" \1 ", line.lower()))
                    .strip()
                    .split(" ")
                )

                curr_tokens = (
                    self.bos_tag + tokens[: self.max_seq_len] + self.eos_tag
                )
                curr_hashings = []
                for j in range(len(curr_tokens)):
                    curr_hashing = murmurhash(curr_tokens[j])
                    curr_hashings.append(curr_hashing[: self.feature_size])
                if label is not None:
                    yield curr_hashings, label
                else:
                    yield curr_hashings
            except:
                continue


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
    train_path: str = "train.csv",
    val_path: str = "test.csv",
    batch_size: int = 32,
    train_size: float = 0.8,
    feature_size: int = 128,
    add_eos_tag: bool = True,
    add_bos_tag: bool = True,
    max_seq_len: int = 256,
    label2index: Dict[str, int] = None,
):

    train_dataset = DummyDataset(
        train_path,
        feature_size=feature_size,
        add_eos_tag=add_eos_tag,
        add_bos_tag=add_bos_tag,
        max_seq_len=max_seq_len,
        label2index=label2index,
    )

    val_dataset = DummyDataset(
        val_path,
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
            num_workers=1,
        ),
        DataLoader(
            val_dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=1,
        ),
    )
