![banner](./banner.png)

<center>
<a href="https://github.com/ChenghaoMou/pytorch-pQRNN"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a> <a href="https://github.com/psf/black/blob/master/LICENSE"><img alt="License: MIT" src="https://black.readthedocs.io/en/stable/_static/license.svg"></a>
</center>

## Note

Because of [this issue](https://github.com/salesforce/pytorch-qrnn/issues/29), [QRNN](https://github.com/salesforce/pytorch-qrnn) is not supported with `torch >= 1.7`. If you want to use a QRNN layer with this repo, please follow the instructions [here](https://github.com/salesforce/pytorch-qrnn) to install `python-qrnn` first with  downgraded `torch <= 1.4`. Otherwise, you can directly run 

```
pip install -r requirements.txt
```

to set up the env.

## Usage

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

-   yelp2(polarity): it will be downloaded w/ datasets(huggingface)
-   yelp5: [json file](https://www.kaggle.com/luisfredgs/hahnn-for-document-classification?select=yelp_reviews.json) should be downloaded to into `data/`
-   toxic: [dataset](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) should be downloaded and unzipped to into `data/`

### Example: Yelp Polarity

    python -W ignore run.py --task yelp2 --b 128 --d 64 --num_layers 4

## Benchmarks(not optimized)

| Model                    | Model Size | Yelp Polarity (error rate) | Yelp-5 (accuracy) | Civil Comments (mean auroc) | Command                                                          |
| ------------------------ | ---------- | -------------------------- | ----------------- | --------------------------- | ---------------------------------------------------------------- |
| ~~PQRNN (this repo)~~<sup>0</sup>    | ~~78K~~    | ~~6.3~~                    | ~~70.4~~          | ~~TODO~~                    | `--b 128 --d 64 --num_layers 4 --rnn_type QRNN`                  |
| PRNN (this repo)         | 90K        | 5.5                        | **70.7**          | 95.57                       | `--b 128 --d 64 --num_layers 1 --rnn_type GRU`                   |
| PTransformer (this repo) | 617K       | 10.8                       | 68              | 86.5                        | `--b 128 --d 64 --num_layers 1 --rnn_type Transformer --nhead 2` |
| PRADO<sup>1</sup>        | 175K       |                            | 65.9              |                             |                                                                  |
| BERT                     | 335M       | **1.81**                   | 70.58             | **98.856**<sup>2</sup>      |                                                                  |
0.  Not supported with `torch >= 1.7`
1.  [Paper](https://www.aclweb.org/anthology/D19-1506.pdf)
2.  Best Kaggle Submission

## Credits

[tensorflow](https://github.com/tensorflow/models/tree/master/research/sequence_projection/prado)

Powered by [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning) and [grid.ai](https://www.grid.ai/)
