# pytorch-pQRNN

Pytorch Implementation of pQRNN.

## Environment

Please follow the instructions [here](https://github.com/salesforce/pytorch-qrnn) to install `python-qrnn`.

```bash
pip install -r requirements.txt
```

## Usage

```bash
Usage: run.py [OPTIONS]

Options:
  --task TEXT           Task to train the model with, currently support
                        `yelp`(yelp polarity) task  [default: yelp]

  --model-type TEXT     Model architecture to use, currently support `pQRNN`
                        [default: pQRNN]

  --b INTEGER           Feature size B from the paper  [default: 256]
  --d INTEGER           d dimention from the paper  [default: 64]
  --num-layers INTEGER  Number of layers for QRNN  [default: 4]
  --batch-size INTEGER  Batch size for the dataloader  [default: 512]
  --dropout FLOAT       Dropout rate  [default: 0.5]
  --lr FLOAT            Learning rate  [default: 0.001]
  --install-completion  Install completion for the current shell.
  --show-completion     Show completion for the current shell, to copy it or
                        customize the installation.

  --help                Show this message and exit.
```

### Example: Yelp Polarity

    python -W ignore run.py --task yelp --b 128 --d 64 --num-layers 4

## Benchmarks(not optimized)

| Model             | Model Size | Yelp Polarity (error rate) |
| ----------------- | ---------- | -------------------------- |
| PQRNN (this repo) | 78K        | 6.3                        |
| BERT large        | 335M       | 1.81                       |

## Credits

[tensorflow](https://github.com/tensorflow/models/tree/master/research/sequence_projection/prado)

Powered by [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning) and [grid.ai](https://www.grid.ai/)
