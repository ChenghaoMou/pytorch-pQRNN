![banner](./banner.png)

![PyPI](https://img.shields.io/pypi/v/pytorch-pqrnn?style=plastic) ![Maintenance](https://img.shields.io/maintenance/yes/2021?style=plastic) ![PyPI - License](https://img.shields.io/pypi/l/pytorch-pqrnn?style=plastic) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4661601.svg)](https://doi.org/10.5281/zenodo.4661601)

## Installation

```bash
# install with pypi
pip install pytorch-pqrnn
# or install locally with poetry
poetry install
```

## Environment

Because of [this issue](https://github.com/salesforce/pytorch-qrnn/issues/29), `pytorch-qrnn` is no longer compatible with pytorch and it is also not actively maintained. If you want to use a QRNN layer in this model, you have install `pytorch-qrnn` with `torch <= 1.4` first.

## Usage

```python
from pytorch_pqrnn.dataset import create_dataloaders
from pytorch_pqrnn.model import PQRNN

model = PQRNN(
  b=128,
  d=96,
  lr=1e-3,
  num_layers=2,
  dropout=0.5,
  output_size=5,
  rnn_type="GRU",
  multilabel=False,
  nhead=2, # used when rnn_type == "Transformer"
)

# Or load the model from your checkpoint
# model = PQRNN.load_from_checkpoint(checkpoint_path="example.ckpt")

# Text data has to be pre-processed with DummyDataset
dataset = DummyDataset(
    df[["text", "label"]].to_dict("records"),
    has_label=True,
    feature_size=128 * 2,
    add_eos_tag=True,
    add_bos_tag=True,
    max_seq_len=512,
    label2index={"pos": 1, "neg": 0},
)

# Explicit train/val loop
# Add model.eval() when necessary
dataloader = create_dataloaders(dataset)
for batch in dataloader:
  # labels could be an empty tensor if has_label is False when creating the dataset. 
  # To change what are included in a batch, feel free to change the collate_fn function
  # in dataset.py
  projections, lengths, labels = batch 
  logits = model.forward(projections)

  # do your magic
```

## CLI Usage

```bash
Usage: run.py [OPTIONS]

Options:
  --task [yelp2|yelp5|toxic]      [default: yelp5]
  --b INTEGER                     [default: 128]
  --d INTEGER                     [default: 96]
  --num_layers INTEGER            [default: 2]
  --batch_size INTEGER            [default: 512]
  --dropout FLOAT                 [default: 0.5]
  --lr FLOAT                      [default: 0.001]
  --nhead INTEGER                 [default: 4]
  --rnn_type [LSTM|GRU|QRNN|Transformer]
                                  [default: GRU]
  --data_path TEXT
  --help                          Show this message and exit.
```

Datasets

-   yelp2(polarity): it will be downloaded w/ huggingface/datasets automatically
-   yelp5: [json file](https://www.kaggle.com/luisfredgs/hahnn-for-document-classification?select=yelp_reviews.json) should be downloaded to into `data_path`
-   toxic: [dataset](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) should be downloaded and unzipped to into `data_path`

### Example: Yelp Polarity

    python -W ignore run.py --task yelp2 --b 128 --d 64 --num_layers 4 --data_path data/

## Benchmarks(not optimized)

| Model                    | Model Size | Yelp Polarity (error rate) | Yelp-5 (accuracy) | Civil Comments (mean auroc) | Command                                                          |
| ------------------------ | ---------- | -------------------------- | ----------------- | --------------------------- | ---------------------------------------------------------------- |
| ~~PQRNN (this repo)~~<sup>0</sup>    | ~~78K~~    | ~~6.3~~                    | ~~70.4~~          | ~~TODO~~                    | `--b 128 --d 64 --num_layers 4 --rnn_type QRNN`                  |
| PRNN (this repo)         | 90K        | 5.5                        | **70.7**          | 95.57                       | `--b 128 --d 64 --num_layers 1 --rnn_type GRU`                   |
| PTransformer (this repo) | 618K       | 10.8                       | 68              | 92.4                        | `--b 128 --d 64 --num_layers 1 --rnn_type Transformer --nhead 8` |
| PRADO<sup>1</sup>        | 175K       |                            | 65.9              |                             |                                                                  |
| BERT                     | 335M       | **1.81**                   | 70.58             | **98.856**<sup>2</sup>      |                                                                  |
0.  Not supported with `torch >= 1.7`
1.  [Paper](https://www.aclweb.org/anthology/D19-1506.pdf)
2.  Best Kaggle Submission

## Credits

- [original tensorflow source code for PRADO](https://github.com/tensorflow/models/tree/master/research/sequence_projection/prado)

- Powered by [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning) and [grid.ai](https://www.grid.ai/)

## Citation

```
@software{chenghao_mou_2021_4661601,
  author       = {Chenghao MOU},
  title        = {ChenghaoMou/pytorch-pQRNN: Add DOI},
  month        = apr,
  year         = 2021,
  publisher    = {Zenodo},
  version      = {0.0.3},
  doi          = {10.5281/zenodo.4661601},
  url          = {https://doi.org/10.5281/zenodo.4661601}
}
```
