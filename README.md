# pytorch-prado
Pytorch Implementation of Prado, and pQRNN (WIP)

## Environment

```bash
pip install -r requirements.txt
```

It's recommended to install [apex](https://github.com/NVIDIA/apex) as well. Otherwise, you should turn off half precision in `run.py` first

## Usage 

Currently, `run.py` only supports training.

```bash
~/pytorch-prado$ python run.py --help
Usage: run.py [OPTIONS]

Options:
  --train-path TEXT     [default: train.ft.txt.bz2]
  --val-path TEXT       [default: test.ft.txt.bz2]
  --b INTEGER           [default: 128]
  --d INTEGER           [default: 96]
  --batch-size INTEGER  [default: 128]
  --install-completion  Install completion for the current shell.
  --show-completion     Show completion for the current shell, to copy it or
                        customize the installation.

  --help                Show this message and exit.
```

I am working on benchmarking and also implementing pQRNN. Stay tuned!

Thanks

## Credits

[tensorflow](https://github.com/tensorflow/models/tree/master/research/sequence_projection/prado)

Powered by [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning) and [grid.ai](https://www.grid.ai/)
