from typing import Any, Dict, List, Tuple, Union

from pathlib import Path

import numpy as np
import pandas as pd
import regex as re
import torch
from datasets import load_dataset
from pytorch_pqrnn.utils import murmurhash
from rich.console import Console
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

REGEX0 = re.compile(r"([\p{S}\p{P}]+)")
REGEX1 = re.compile(r" +")

console = Console()


class DummyDataset(Dataset):
    def __init__(
        self,
        x: Union[pd.DataFrame, List[Dict[str, Any]]],
        has_label: bool = True,
        feature_size: int = 512,
        add_eos_tag: bool = True,
        add_bos_tag: bool = True,
        max_seq_len: int = 256,
        label2index: Dict[str, int] = None,
    ):
        """Convert the dataset into a dummy torch dataset.

        Parameters
        ----------
        x : Union[pd.DataFrame, List[Dict[str, Any]]]
            A dataframe or a list of dicts
        has_label : bool, optional
            Whether the dataframe or each dict has a label key/column, by default True
        feature_size : int, optional
            Dimension of the hash, the output dimension would be half of the hash size(2b -> b), by default 512
        add_eos_tag : bool, optional
            Add a special eos tag, by default True
        add_bos_tag : bool, optional
            Add a special bos tag, by default True
        max_seq_len : int, optional
            Maximum sequence length, by default 256
        label2index : Dict[str, int], optional
            Mapping that converts labels into indices, by default None

        Examples
        --------
        >>> data = [{'text': 'Hello world!', 'label': 'positive'}]
        >>> dataset = DummyDataset(data, has_label=True, feature_size=512, label2index={'positive': 1, 'negative': 0})
        >>> hash, label = dataset[0]
        >>> assert label == 1, label
        >>> assert len(hash[0]) == 256, len(hash[0])
        """
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


def collate_fn(examples: List[Any]) -> Tuple[torch.Tensor, ...]:
    """Batching examples.

    Parameters
    ----------
    examples : List[Any]
        List of examples

    Returns
    -------
    Tuple[torch.Tensor, ...]
        Tuple of hash tensor, length tensor, and label tensor
    """

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
    task: str = "yelp2",
    batch_size: int = 32,
    feature_size: int = 128,
    add_eos_tag: bool = True,
    add_bos_tag: bool = True,
    max_seq_len: int = 256,
    label2index: Dict[str, int] = None,
    data_path: Union[str, Path] = "data",
) -> Tuple[DataLoader, DataLoader]:
    """Create train and eval dataloaders.

    Parameters
    ----------
    task : str, optional
        Name from predefined tasks, by default "yelp2"
    batch_size : int, optional
        Size of the batch, by default 32
    feature_size : int, optional
        Dimension of the features, by default 128
    add_eos_tag : bool, optional
        Add a special eos tag, by default True
    add_bos_tag : bool, optional
        Add a special bos tag, by default True
    max_seq_len : int, optional
        Maximum sequence length, by default 256
    label2index : Dict[str, int], optional
        Mapping that converts labels to indices, by default None
    data_path: str, optional
        Path to the data files

    Returns
    -------
    Tuple[DataLoader, DataLoader]
        Train and eval dataloaders

    Raises
    ------
    Exception
        Unsupported task
    """

    data_path = Path(data_path)

    if task == "yelp2":
        dataset = load_dataset("yelp_polarity")
    elif task == "yelp5":
        data = pd.read_json(data_path / "yelp_reviews.json", lines=True)
        data["label"] = data["stars"] - 1
        data["label"] = data["label"].astype(int)
        train, val = train_test_split(
            data, test_size=0.1, stratify=data["label"]
        )
        dataset = {
            "train": train.to_dict("records"),
            "test": val.to_dict("records"),
        }
    elif task == "toxic":
        train = pd.read_csv(data_path / "train.csv")
        labels = pd.read_csv(data_path / "test_labels.csv")
        test = pd.read_csv(data_path / "test.csv")
        labels["id"] = labels["id"].astype(str)
        test["id"] = test["id"].astype(str)
        test = test.merge(labels)
        test = test[test["toxic"] != -1]

        train["text"] = train["comment_text"]
        train["label"] = train[
            [
                "toxic",
                "severe_toxic",
                "obscene",
                "threat",
                "insult",
                "identity_hate",
            ]
        ].values.tolist()

        test["text"] = test["comment_text"]
        test["label"] = test[
            [
                "toxic",
                "severe_toxic",
                "obscene",
                "threat",
                "insult",
                "identity_hate",
            ]
        ].values.tolist()

        dataset = {
            "train": train[["text", "label"]].to_dict("records"),
            "test": test[["text", "label"]].to_dict("records"),
        }
    else:
        raise Exception(f"Unsupported task: {task} VS. {{yelp2, yelp5}}")

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
