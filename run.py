import os

import pytorch_lightning as pl
import torch
import typer
from dataset import create_dataloaders
from model import PRADO
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from rich.console import Console
from typer import Typer

app = Typer()
console = Console()


@app.command()
def train(
    train_path: str = typer.Option("train.ft.txt.bz2"),
    val_path: str = typer.Option("test.ft.txt.bz2"),
    B: int = typer.Option(128),
    d: int = typer.Option(96),
    batch_size: int = typer.Option(128),
):

    model = PRADO(B=B, d=d)

    train_dataloader, dev_dataloader = create_dataloaders(
        train_path,
        val_path,
        batch_size=batch_size,
        feature_size=B,
        label2index={"__label__1": 0, "__label__2": 1},
    )

    trainer = pl.Trainer(
        logger=pl_loggers.TensorBoardLogger("lightning_logs", log_graph=False),
        early_stop_callback=EarlyStopping(monitor="val_loss"),
        deterministic=True,
        val_check_interval=2048,
        gpus=[0] if torch.cuda.is_available() else None,
        precision=16,
        amp_level="O2",
        amp_backend="apex",
    )

    trainer.fit(model, train_dataloader, dev_dataloader)


if __name__ == "__main__":

    app()
