import os

import pytorch_lightning as pl
import torch
import typer
from dataset import create_dataloaders
from model import PQRNN
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    model_checkpoint,
)
from rich.console import Console
from typer import Typer

app = Typer()
console = Console()


@app.command()
def train(
    task: str = typer.Option(
        "yelp",
        allow_dash=True,
        help="Task to train the model with, currently support `yelp`(yelp polarity) task",
    ),
    model_type: str = typer.Option(
        "pQRNN",
        allow_dash=True,
        help="Model architecture to use, currently support `pQRNN`",
    ),
    b: int = typer.Option(
        256, allow_dash=True, help="Feature size B from the paper"
    ),
    d: int = typer.Option(
        64, allow_dash=True, help="d dimention from the paper"
    ),
    num_layers: int = typer.Option(
        4, allow_dash=True, help="Number of layers for QRNN"
    ),
    batch_size: int = typer.Option(512, help="Batch size for the dataloader"),
    dropout: float = typer.Option(0.5, allow_dash=True, help="Dropout rate"),
    lr: float = typer.Option(1e-3, allow_dash=True, help="Learning rate"),
):

    train_dataloader, dev_dataloader = create_dataloaders(
        task,
        batch_size=batch_size,
        feature_size=b * 2,
        label2index=None,
    )

    model = PQRNN(b=b, d=d, lr=lr, num_layers=num_layers, dropout=dropout)

    trainer = pl.Trainer(
        logger=pl_loggers.TensorBoardLogger("lightning_logs", log_graph=False),
        early_stop_callback=EarlyStopping(monitor="val_loss", patience=5),
        checkpoint_callback=ModelCheckpoint(
            "./checkpoints/", monitor="val_loss"
        ),
        deterministic=True,
        val_check_interval=0.2,
        gpus=[0] if torch.cuda.is_available() else None,
        gradient_clip_val=1.0,
    )

    trainer.fit(model, train_dataloader, dev_dataloader)


if __name__ == "__main__":

    app()
