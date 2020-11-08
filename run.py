import os

import pytorch_lightning as pl
import torch
import typer
from dataset import create_dataloaders
from model import PQRNN, PRADO
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from rich.console import Console
from typer import Typer

app = Typer()
console = Console()


@app.command()
def train(
    task: str = typer.Option("yelp", allow_dash=True),
    B: int = typer.Option(512, allow_dash=True),
    d: int = typer.Option(96, allow_dash=True),
    batch_size: int = typer.Option(64),
    lr: float = typer.Option(1e-3, allow_dash=True),
):

    model = PQRNN(B=B, d=d, lr=lr)

    train_dataloader, dev_dataloader = create_dataloaders(
        task,
        batch_size=batch_size,
        feature_size=B * 2,
        label2index=None,
    )

    trainer = pl.Trainer(
        logger=pl_loggers.TensorBoardLogger("lightning_logs", log_graph=False),
        early_stop_callback=EarlyStopping(monitor="val_loss"),
        deterministic=True,
        val_check_interval=0.1,
        gpus=[0] if torch.cuda.is_available() else None,
        # precision=16,
        # amp_level="O2",
        # amp_backend="apex",
        # overfit_batches=10,
        gradient_clip_val=1.0,
    )

    trainer.fit(model, train_dataloader, dev_dataloader)


if __name__ == "__main__":

    app()
