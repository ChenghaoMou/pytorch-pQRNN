from typing import List

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    import pytorch_lightning as pl
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from pytorch_lightning.metrics.functional.classification import (
        accuracy,
        f1_score,
    )
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from torchqrnn import QRNN

_NGRAM_INFO = [
    {"name": "unigram", "padding": 0, "kernel_size": [1, 1], "mask": None},
    {"name": "bigram", "padding": 1, "kernel_size": [2, 1], "mask": None},
    {"name": "trigram", "padding": 2, "kernel_size": [3, 1], "mask": None},
    {
        "name": "bigramskip1",
        "padding": 2,
        "kernel_size": [3, 1],
        "mask": [[[[1]]], [[[0]]], [[[1]]]],
    },
    {
        "name": "bigramskip2",
        "padding": 3,
        "kernel_size": [4, 1],
        "mask": [[[[1]]], [[[0]]], [[[0]]], [[[1]]]],
    },
    {"name": "fourgram", "padding": 3, "kernel_size": [4, 1], "mask": None},
    {"name": "fivegram", "padding": 4, "kernel_size": [5, 1], "mask": None},
]


# class PRADO(pl.LightningModule):
#     def __init__(
#         self,
#         b: int = 512,
#         d: int = 64,
#         heads: List[int] = None,
#         fc_sizes: List[int] = None,
#         output_size: int = 2,
#         lr: float = 0.025,
#         dropout: float = 0.5,
#     ):
#         super().__init__()
#         if heads is None:
#             heads = [0, 64, 64, 0, 0]
#         if fc_sizes is None:
#             fc_sizes = [128, 64]  # no original default values

#         self.b = b
#         self.d = d
#         self.heads = heads
#         self.fc_sizes = fc_sizes
#         self.lr = lr
#         self.output_size = output_size
#         self.tanh = nn.Hardtanh()

#         self.linear_key = nn.Linear(b, d)
#         self.linear_value = nn.Linear(b, d)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(dropout)

#         convs = {}
#         for info, channels in zip(_NGRAM_INFO, self.heads):
#             if not channels:
#                 continue
#             # ? weight_mask = info["mask"] pytorch does not have weight mask
#             kernel_size = info["kernel_size"]
#             convs[info["name"]] = nn.Conv2d(
#                 1,
#                 channels,
#                 kernel_size,
#             )

#         self.convs = nn.ModuleDict(convs)

#         layers = []
#         for x, y in zip([sum(self.heads)] + fc_sizes[:-1], fc_sizes):
#             layers.append(nn.Linear(x, y))
#             layers.append(nn.ReLU())

#         layers.append(nn.Linear(fc_sizes[-1], output_size))

#         self.output = nn.ModuleList(layers)

#         self.loss = nn.CrossEntropyLoss()

#     def forward(self, projection, seq_lengths):
#         features = self.tanh(projection)
#         batch_size, max_seq_len, _ = features.shape
#         values, keys = (
#             self.dropout(self.relu(self.linear_key(features))),
#             self.dropout(self.relu(self.linear_value(features))),
#         )
#         valid_step_mask = torch.arange(
#             max_seq_len, device=features.device, dtype=features.dtype
#         ).expand(batch_size, max_seq_len) < seq_lengths.unsqueeze(1)
#         valid_step_mask = valid_step_mask.unsqueeze(dim=-1)
#         mask = valid_step_mask.unsqueeze(dim=2).unsqueeze(dim=2).squeeze(dim=-1)

#         keys = valid_step_mask * keys  # [batch_size, max_seq_length, d]
#         values = valid_step_mask * values  # [batch_size, max_seq_length, d]

#         keys = keys.unsqueeze(dim=1)  # [batch_size, 1, max_seq_length, d]
#         values = values.unsqueeze(dim=1)  # [batch_size, 1, max_seq_length, d]

#         multi_head_predictions = []
#         for head_type, _ in enumerate(self.heads):

#             info = _NGRAM_INFO[head_type]
#             if info["name"] not in self.convs:
#                 continue

#             pad = info["padding"]
#             paddings = [0, 0, 0, pad, 0, 0, 0, 0]
#             key_tensor = F.pad(keys, paddings) if pad != 0 else keys
#             value_tensor = F.pad(values, paddings) if pad != 0 else values
#             key_tensor = self.convs[info["name"]](key_tensor)
#             value_tensor = self.convs[info["name"]](value_tensor)

#             key_tensor = key_tensor.permute([0, 2, 3, 1])
#             value_tensor = value_tensor.permute([0, 2, 3, 1])
#             key_tensor = key_tensor * mask + (~mask) * -100
#             value_tensor = value_tensor * mask + (~mask) * 0

#             channels = key_tensor.shape[-1]

#             key_tensor, value_tensor = (
#                 key_tensor.reshape([batch_size, -1, channels]),
#                 value_tensor.reshape([batch_size, -1, channels]),
#             )
#             key_tensor = key_tensor.transpose(
#                 1, 2
#             )  # [batch_size, channels, max_seq_length * B]
#             value_tensor = value_tensor.transpose(
#                 1, 2
#             )  # [batch_size, channels, max_seq_length * B]
#             attention = F.softmax(key_tensor, dim=-1)
#             output = torch.sum(
#                 attention * value_tensor, dim=-1
#             )  # [batch_size, channels]
#             multi_head_predictions.append(output)

#         multi_head_predictions = torch.cat(multi_head_predictions, axis=1)
#         logits = multi_head_predictions
#         for layer in self.output:
#             logits = layer(logits)
#         return logits

#     def training_step(self, batch, batch_idx):
#         projection, seq_lengths, labels = batch
#         logits = self.forward(projection, seq_lengths)
#         self.log("loss", self.loss(logits, labels).detach().cpu().item())
#         return {"loss": self.loss(logits, labels)}

#     def validation_step(self, batch, batch_idx):
#         projection, seq_lengths, labels = batch
#         logits = self.forward(projection, seq_lengths)

#         return {"logits": logits, "labels": labels}

#     def validation_epoch_end(self, outputs):

#         logits = torch.cat([o["logits"] for o in outputs], dim=0)
#         labels = torch.cat([o["labels"] for o in outputs], dim=0)
#         self.log(
#             "val_f1",
#             f1_score(
#                 torch.argmax(logits, dim=1), labels, class_reduction="macro"
#             )
#             .detach()
#             .cpu()
#             .item(),
#             prog_bar=True,
#         )
#         self.log(
#             "val_acc",
#             accuracy(torch.argmax(logits, dim=1), labels).detach().cpu().item(),
#             prog_bar=True,
#         )
#         self.log(
#             "val_loss",
#             self.loss(logits, labels).detach().cpu().item(),
#             prog_bar=True,
#         )

#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
#         return optimizer


class PQRNN(pl.LightningModule):
    def __init__(
        self,
        b: int = 512,
        d: int = 64,
        num_layers: int = 4,
        fc_sizes: List[int] = None,
        output_size: int = 2,
        lr: float = 0.025,
        dropout: float = 0.5,
    ):
        super().__init__()
        if fc_sizes is None:
            fc_sizes = [128, 64]

        self.hparams = {
            "b": b,
            "d": d,
            "fc_size": fc_sizes,
            "lr": lr,
            "output_size": output_size,
            "dropout": dropout,
        }

        layers: List[nn.Module] = []
        for x, y in zip([d] + fc_sizes, fc_sizes + [output_size]):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(x, y))

        self.tanh = nn.Hardtanh()
        self.qrnn = QRNN(b, d, num_layers=num_layers, dropout=dropout)
        self.output = nn.ModuleList(layers)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, projection):
        features = self.tanh(projection)
        features = features.transpose(0, 1)
        output, _ = self.qrnn(features)
        output = output.transpose(0, 1)
        logits = torch.mean(output, dim=1)
        for layer in self.output:
            logits = layer(logits)
        return logits

    def training_step(self, batch, batch_idx):
        projection, _, labels = batch
        logits = self.forward(projection)
        self.log("loss", self.loss(logits, labels).detach().cpu().item())
        return {"loss": self.loss(logits, labels)}

    def validation_step(self, batch, batch_idx):
        projection, _, labels = batch
        logits = self.forward(projection)

        return {"logits": logits, "labels": labels}

    def validation_epoch_end(self, outputs):

        logits = torch.cat([o["logits"] for o in outputs], dim=0)
        labels = torch.cat([o["labels"] for o in outputs], dim=0)
        self.log(
            "val_f1",
            f1_score(
                torch.argmax(logits, dim=1), labels, class_reduction="macro"
            )
            .detach()
            .cpu()
            .item(),
            prog_bar=True,
        )
        self.log(
            "val_acc",
            accuracy(torch.argmax(logits, dim=1), labels).detach().cpu().item(),
            prog_bar=True,
        )
        self.log(
            "val_loss",
            self.loss(logits, labels).detach().cpu().item(),
            prog_bar=True,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])
        scheduler = ReduceLROnPlateau(optimizer, "min")
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }
