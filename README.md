---   
<div align="center">    
 
# Generalizing Lottery Tickets   

 [![DOI](https://zenodo.org/badge/224994704.svg)](https://zenodo.org/badge/latestdoi/224994704)  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>
 

 
## Description   
This repository contains PyTorch code to replicate the experiments given in NeurIPS 2019 paper 

[___"One ticket to win them all: generalizing lottery ticket initializations across datasets and optimizers"___](https://arxiv.org/abs/1906.02773v2)

As finding the _winning lottery tickets_ is computationally expensive, we also open-source winning tickets (pretrained and pruned models) we generated during our experiments. Link : [Winning Tickets](https://drive.google.com/drive/folders/1Nd-J4EwmgWbUARYaqe9iCF6efEFf9S2P?usp=sharing)

I have included the SNIP pruning method that was introduced in a conference paper at ICLR 2019

[___"SNIP: Single-shot Network Pruning based on Connection Sensitivity"___](https://arxiv.org/abs/1810.02340)

in order to compare the two advanced pruning methods.

## How to Setup    
```bash
# clone project   
git clone https://github.com/67Samuel/Generalizing-Lottery-Tickets.git

# install all dependencies   
cd Generalizing-Lottery-Tickets   
# set up virtual env with pip3
pip3 install -r requirements.txt
# or anaconda
conda env create -f environment.yml
conda activate gen-lt-env
```

## How to Run
There are 6 files in ```src``` folder:
- train.py             : Use to train the neural network and find the winning ticket
- snip_train.py        : Use to prune and train the neural network using SNIP
- test.py              : Use to test the accuracy of the trained model
- iterative_pruning.py : Use to iteratively prune the model using the lottery ticket hypothesis
- iterative_snip.py    : Use to iteratively prune the model using SNIP (for comparison purposes only, note that SNIP was designed as a one-shot pruning method)
- utils.py             : Contains helper functions used in scripts mentioned above

To support more datasets and architectures, we need to add necessary code to utils.py

### Using train.py / snip_train.py
##### Mandatory arguments:
- --architecture : To specify the neural network architecture (vgg19, resnet50 or alexnet)
- --dataset      : The dataset to train on (cifar10, cifar100, fashionmnist, svhn, cifar10a, cifar10b)
- --wandb        : Set to True to log results to wandb
##### Optional arguments to note:
- --batch_size : To set the batch size while training
- --optimizer  : The optimizer to use for training (sgd and adam). sgd used by default
- --model_saving_path : Path to directory where trained model is saved. (default is ./)
- --snip       : Percentage of model you want to prune using SNIP (default is 50%)
- --entity     : Entity (username) of wandb account (must use if wandb is True)
- --project    : Wandb project to log results to
- --run_name   : Run name to log to wandb

Models will be saved after loss decreases below 0.3, in intervals of num_epochs/10. VGG19 will be saved for every 16<sup>th</sup> epoch. FOr Resnet50, the model will be saved for every 9<sup>th</sup> epoch. For our experiments, while pruning, we reinitialize the model with weights after epoch 2 (late resetting of 1).
```bash
# source folder
cd Generalizing-Lottery-Ticket/src   

# run train.py
python train.py --architecture=resnet50 --dataset=cifar100 --wandb --entity=67Samuel

# run snip_train.py
python snip_train.py --architecture=resnet50 --dataset=cifar100 --snip=70 --wandb --entity=67Samuel --project=SNIP_trained --project=project_name --run_name=run_name
```

### Using iterative_pruning.py
##### Mandatory arguments:
- --architecture : To specify the neural network architecture (vgg19 and resnet50)
- --target_dataset      : The dataset to train on (cifar10, cifar100, fashionmnist, svhn, cifar10a, cifar10b)
- --source_dataset      : The dataset using which winning ticket initialization was found (cifar10, cifar100, fashionmnist, svhn, cifar10a, cifar10b)
- --wandb       : Set to True to log results to wandb
- --init_path   : Path to winning initialization model

##### Optional arguments:
- --batch_size : To set the batch size while training
- --optimizer  : The optimizer to use for training (sgd and adam). sgd used by default
- --seed : To set the ranodm seed
- --model_saving_path : Path to directory where pruned model will be saved (default is ./)
- --entity     : Entity (username) of wandb account (must use if wandb is True)
- --project    : Wandb project to log results to
- --run_name   : Run name to log to wandb

The script will run 30 pruning iterations which will prune away 99.9% of the weights. The trained and pruned model will be saved at end of each pruning iteration

```bash
# source folder
cd Generalizing-Lottery-Ticket/src   

# run iterative_pruning.py and iterative_snip.py the same way
python iterative_pruning.py --architecture=resnet50 --source_dataset=cifar100 --target_dataset=cifar100 --init_path=./ --model_saving_path=<path-to-dir-where-models-are-to-be-stored> --wandb --entity=67Samuel --project=project_name --run_name=run_name
```

### Using test.py
##### Mandatory arguments:
- --architecture : To specify the neural network architecture (vgg19 and resnet50)
- --dataset      : The dataset to train on (cifar10, cifar100, fashionmnist, svhn, cifar10a, cifar10b)
- --model_path   : The path to moedl whose accuracy needs to be evaluated.
- --wandb       : Set to True to log results to wandb

##### Optional arguments:
- --batch-size : To set the batch size while training
- --entity     : Entity (username) of wandb account (must use if wandb is True)
- --project    : Wandb project to log results to
- --run_name   : Run name to log to wandb

Running this script will print the _Fraction of pruned weights_ in the model and the _Test Accuracy_. 
```bash
# source folder
cd Generalizing-Lottery-Ticket/src   

# run test.py
python test.py --architecture=resnet50 --dataset=cifar10 --model-path=<path-to-model> --wandb --entity=67Samuel
```

### Results   
The results of the replicated experiments can be found in plots folder.
  

### Citation 
If you use this repository, kindly cite the reproducibility report and the original paper. The bibtex is given below.
```
@article{Gohil:2020,
  author = {Gohil, Varun and Narayanan, S. Deepak and Jain, Atishay},
  title = {{[Re] One ticket to win them all: generalizing lottery ticket initializations across datasets and optimizers}},
  journal = {ReScience C},
  year = {2020},
  month = may,
  volume = {6},
  number = {2},
  pages = {{#4}},
  doi = {10.5281/zenodo.3818619},
  url = {https://zenodo.org/record/3818619/files/article.pdf},
  code_url = {https://github.com/varungohil/Generalizing-Lottery-Tickets},
  code_doi = {10.5281/zenodo.3700320},
  code_swh = {swh:1:dir:8a9e53bc8a9028428bbad6a4e77ae3fedae49d30},
  data_url = {},
  data_doi = {},
  review_url = {https://openreview.net/forum?id=SklFHaqG6S},
  type = {Replication},
  language = {Python},
  domain = {NeurIPS 2019 Reproducibility Challenge},
  keywords = {lottery ticket hypothesis, pytorch}
}

@incollection{NIPS2019_8739,
title = {One ticket to win them all: generalizing lottery ticket initializations across datasets and optimizers},
author = {Morcos, Ari and Yu, Haonan and Paganini, Michela and Tian, Yuandong},
booktitle = {Advances in Neural Information Processing Systems 32},
editor = {H. Wallach and H. Larochelle and A. Beygelzimer and F. d\textquotesingle Alch\'{e}-Buc and E. Fox and R. Garnett},
pages = {4932--4942},
year = {2019},
publisher = {Curran Associates, Inc.},
url = {http://papers.nips.cc/paper/8739-one-ticket-to-win-them-all-generalizing-lottery-ticket-initializations-across-datasets-and-optimizers.pdf}
}

```  
### Core Contributors
[Varun Gohil](https://varungohil.github.io), [S. Deepak Narayanan](https://sdeepaknarayanan.github.io), [Atishay Jain](https://github.com/AtishayJain-ML)

### Development
We have a new branch ```dev``` in which pull requests are welcome. We will merge them after reviewing. 

### Contributors
[fcorencot](https://github.com/fcorencoret)
