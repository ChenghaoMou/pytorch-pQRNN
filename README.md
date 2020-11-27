# pytorch-pQRNN

Pytorch Implementation of pQRNN.

## Environment

Please follow the instructions [here](https://github.com/salesforce/pytorch-qrnn) to install `python-qrnn` first if you want to use QRNN. Because of the cuda-specific implementation of QRNN, pQRNN model cannot run on a CPU-only machine.

```bash
pip install -r requirements.txt
```

## Usage

```bash
Usage: run.py [OPTIONS]

Options:
  --task [yelp2|yelp5]        [default: yelp5]
  --b INTEGER                 [default: 128]
  --d INTEGER                 [default: 96]
  --num_layers INTEGER        [default: 4]
  --batch_size INTEGER        [default: 512]
  --dropout FLOAT             [default: 0.5]
  --lr FLOAT                  [default: 0.001]
  --rnn_type [LSTM|GRU|QRNN]  [default: LSTM]
  --help                      Show this message and exit.
```

Datasets

-   yelp(polarity): it will be downloaded w/ datasets(huggingface)
-   yelp-5: [json file](https://www.kaggle.com/luisfredgs/hahnn-for-document-classification?select=yelp_reviews.json) should be downloaded to into `data/`

### Example: Yelp Polarity

    python -W ignore run.py --task yelp --b 128 --d 64 --num-layers 4

## Benchmarks(not optimized)

| Model                                                         | Model Size | Yelp Polarity (error rate) | Yelp-5 (accuracy) |                    Command                    |
| ------------------------------------------------------------- | :--------: | :------------------------: | :---------------: | :-------------------------------------------: |
| PQRNN (this repo)                                             |     78K    |             6.3            |       70.4\*      | --b 128 --d 64 --num-layers 4 --rnn_type QRNN |
| PRNN (this repo)                                              |     90K    |            TODO            |        TODO       |  --b 128 --d 64 --num_layers 1 --rnn_type GRU |
| PTransformer (this repo)                                      |    TODO    |            TODO            |        TODO       |                                               |
| PRADO([paper](https://www.aclweb.org/anthology/D19-1506.pdf)) |    175K    |                            |        65.9       |                                               |
| BERT                                                          |    335M    |            1.81            |       70.58       |                                               |

-   tested on 10% of the data

## Credits

[tensorflow](https://github.com/tensorflow/models/tree/master/research/sequence_projection/prado)

Powered by [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning) and [grid.ai](https://www.grid.ai/)
