# Hover-Net

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1QdMKJdhCyy9P9bwiLpcB5MhlkxRNnCYj?usp=sharing)

This project attempts to reproduce the architecture specified in the paper  
Graham, Simon, et al. ["Hover-Net: Simultaneous segmentation and classification of nuclei in multi-tissue histology images."](https://arxiv.org/abs/1812.06499) Medical Image Analysis 58 (2019): 101563  
in PyTorch and is currently a **work in progress**.

The most up-to-date version is currently in the Colab notebook. 

Use the steps below to get the code on your local machine.

## Installation

Start by cloning the repository

```bash
git clone https://github.com/rshwndsz/hover-net.git
```

If you don't already have an environment with PyTorch 1.x, it's better to create a new conda environment with Python 3.6+.

Install `conda-env` if you haven't already.

```bash
conda install -c conda conda-env
```

To create a new environment and install required packages available on conda, run

```console
$ conda env create --file environment.yml 
Fetching package metadata: ...
Solving package specifications: .Linking packages ...
[      COMPLETE      ] |#################################################| 100%
#
# To activate this environment, use:
# $ source activate hovernet-env
#
# To deactivate this environment, use:
# $ source deactivate
#
```

Move into your new environment

```console
$ source activate hovernet-env
(hovernet-env) $
```

Install PyTorch and the CUDA Toolkit based on your local configuration. Get the command for the installation from [the official website](https://pytorch.org/).

The command for a Linux system with a GPU and CUDA 10.1 installed is given below.

```console
(hovernet-env) $ conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```

Install packages not available through conda using pip

```console
(hovernet-env) $ pip install -r requirements.txt
```

## Setup

Get the "Multi-organ dataset by Neeraj Kumar et.al" from [https://monuseg.grand-challenge.org/Data/](https://monuseg.grand-challenge.org/Data/) or download a cropped (to 256x256) version using

```console
(hovernet-env) $ ./install.sh
```

or download your own dataset into `dataset/raw/`.

Edit the dataset APIs in `hovernet/data.py` and `test.py` as required.

## Training

To train the model defined in `hovernet/model.py` using default parameters run

```console
(hovernet-env) $ python train.py
```

This trains the model for 10 epochs and saves the best model (based on validation loss) in `checkpoints/model-saved.pth`

You can specify a lot of parameters as command line arguments.
To find out which parameters can be provided run

```console
(hovernet-env) $ python train.py --help
```

## Testing

To test the model run

```console
(hovernet-env) $ python test.py
```
