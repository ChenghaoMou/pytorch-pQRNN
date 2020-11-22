# pytorch-pQRNN

Pytorch Implementation of pQRNN.

## Environment

Please follow the instructions [here](https://github.com/salesforce/pytorch-qrnn) to install `python-qrnn` first. Because of the cuda-specific implementation of QRNN, this model cannot run on a CPU-only machine.

```bash
pip install -r requirements.txt
```

## Usage

```bash
Usage: run.py [OPTIONS]

Options:
  --task TEXT           Task to train the model with, currently support
                        `yelp`(yelp polarity) and `yelp-5` tasks  [default:
                        yelp]

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

Datasets

-   yelp(polarity): it will be downloaded w/ datasets(huggingface)
-   yelp-5: [json file](https://www.kaggle.com/luisfredgs/hahnn-for-document-classification?select=yelp_reviews.json) should be downloaded to into `data/`

### Example: Yelp Polarity

    python -W ignore run.py --task yelp --b 128 --d 64 --num-layers 4

## Benchmarks(not optimized)

| Model                                                         | Model Size | Yelp Polarity (error rate) | Yelp-5 (accuracy) |
| ------------------------------------------------------------- | :--------: | :------------------------: | :---------------: |
| PQRNN (this repo)                                             |     78K    |             6.3            |       70.4\*      |
| PRADO([paper](https://www.aclweb.org/anthology/D19-1506.pdf)) |    175K    |                            |        65.9       |
| BERT                                                          |    335M    |            1.81            |       70.58       |

-   tested on 10% of the data

## Credits

[tensorflow](https://github.com/tensorflow/models/tree/master/research/sequence_projection/prado)

Powered by [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning) and [grid.ai](https://www.grid.ai/)
