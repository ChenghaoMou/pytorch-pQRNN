import click
import pytorch_lightning as pl
import torch
from dataset import create_dataloaders
from model import PQRNN
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from rich.console import Console

console = Console()


@click.command()
@click.option(
    "--task",
    type=click.Choice(["yelp2", "yelp5", "toxic"], case_sensitive=False),
    default="yelp5",
    show_default=True,
)
@click.option("--b", type=int, default=128, show_default=True)
@click.option("--d", type=int, default=96, show_default=True)
@click.option("--num_layers", type=int, default=2, show_default=True)
@click.option("--batch_size", type=int, default=512, show_default=True)
@click.option("--dropout", type=float, default=0.5, show_default=True)
@click.option("--lr", type=float, default=1e-3, show_default=True)
@click.option("--nhead", type=int, default=4, show_default=True)
@click.option(
    "--rnn_type",
    type=click.Choice(
        ["LSTM", "GRU", "QRNN", "Transformer"], case_sensitive=False
    ),
    default="GRU",
    show_default=True,
)
@click.option("--data_path", type=str, default="data")
def train(
    task: str,
    b: int,
    d: int,
    num_layers: int,
    batch_size: int,
    dropout: float,
    lr: float,
    nhead: int,
    rnn_type: str,
    data_path: str,
):

    train_dataloader, dev_dataloader = create_dataloaders(
        task,
        batch_size=batch_size,
        feature_size=b * 2,
        label2index=None,
        data_path=data_path,
    )
    num_classes = {"yelp2": 2, "yelp5": 5, "toxic": 6}.get(task, 2)

    model = PQRNN(
        b=b,
        d=d,
        lr=lr,
        num_layers=num_layers,
        dropout=dropout,
        output_size=num_classes,
        rnn_type=rnn_type,
        multilabel=task == "toxic",
        nhead=nhead,
    )

    trainer = pl.Trainer(
        logger=pl_loggers.TensorBoardLogger("lightning_logs", log_graph=False),
        callbacks=[EarlyStopping(monitor="val_loss", patience=10)],
        checkpoint_callback=ModelCheckpoint(
            "./checkpoints/", monitor="val_loss"
        ),
        min_epochs=2,
        deterministic=True,
        val_check_interval=0.2,
        gpus=[0] if torch.cuda.is_available() else None,
        gradient_clip_val=1.0,
    )

    trainer.fit(model, train_dataloader, dev_dataloader)


if __name__ == "__main__":

    train()
