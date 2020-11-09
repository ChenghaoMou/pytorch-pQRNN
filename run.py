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
    task: str = typer.Option("yelp", allow_dash=True),
    model_type: str = typer.Option("pQRNN", allow_dash=True),
    b: int = typer.Option(256, allow_dash=True),
    d: int = typer.Option(64, allow_dash=True),
    batch_size: int = typer.Option(512),
    lr: float = typer.Option(1e-3, allow_dash=True),
):

    train_dataloader, dev_dataloader = create_dataloaders(
        task,
        batch_size=batch_size,
        feature_size=b * 2,
        label2index=None,
    )

    model = PQRNN(b=b, d=d, lr=lr)

    trainer = pl.Trainer(
        logger=pl_loggers.TensorBoardLogger("lightning_logs", log_graph=False),
        early_stop_callback=EarlyStopping(monitor="val_loss", patience=5),
        checkpoint_callback=ModelCheckpoint(
            "./checkpoints", monitor="val_loss"
        ),
        deterministic=True,
        val_check_interval=0.2,
        gpus=[0] if torch.cuda.is_available() else None,
        gradient_clip_val=1.0,
    )

    trainer.fit(model, train_dataloader, dev_dataloader)


if __name__ == "__main__":

    app()
