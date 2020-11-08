# pytorch-prado
Pytorch Implementation of Prado, and pQRNN

## Environment

```bash
pip install -r requirements.txt
```

## Note

- If you want to use pQRNN, please follow the instructions [here](https://github.com/salesforce/pytorch-qrnn) to install `python-qrnn`
- It's recommended to install [apex](https://github.com/NVIDIA/apex) as well. Otherwise, you should turn off half precision in `run.py` first

## Usage 

Currently, `run.py` only supports training.

```bash
Usage: run.py [OPTIONS]

Options:
  --task TEXT           [default: yelp]
  --b INTEGER           [default: 512]
  --d INTEGER           [default: 96]
  --batch-size INTEGER  [default: 64]
  --lr FLOAT            [default: 0.001]
  --install-completion  Install completion for the current shell.
  --show-completion     Show completion for the current shell, to copy it or
                        customize the installation.

  --help                Show this message and exit.
```

### Yelp
```
python -W ignore run.py --task yelp --b 128 --d 64
```

## Benchmarks(not optimized)

| Model             	| Model Size 	| Yelp (error rate) 	      |
|-------------------	|------------	|--------------------------	|
| PQRNN (this repo) 	| 53K        	| 8.4                     	|
| BERT large        	| 335M       	| 1.81                     	|

## Credits

[tensorflow](https://github.com/tensorflow/models/tree/master/research/sequence_projection/prado)

Powered by [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning) and [grid.ai](https://www.grid.ai/)
