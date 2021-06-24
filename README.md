[![Build Status](https://travis-ci.com/vturrisi/solo-learn.svg?branch=main)](https://travis-ci.com/vturrisi/solo-learn)
[![codecov](https://codecov.io/gh/vturrisi/solo-learn/branch/main/graph/badge.svg?token=WLU9UU17XZ)](https://codecov.io/gh/vturrisi/solo-learn)

# solo-learn

A library of self-supervised methods for unsupervised visual representation learning powered by PyTorch Lightning.
We aim at providing SOTA self-supervised methods in a comparable environment while, at the same time, implementing training tricks.
While the library is self contained, it is possible to use the models outside of solo-learn environment.

---

## Methods available:
* [Barlow Twins](https://arxiv.org/abs/2103.03230)
* [BYOL](https://arxiv.org/abs/2006.07733)
* [MoCo-V2](https://arxiv.org/abs/2003.04297)
* [NNCLR](https://arxiv.org/abs/2104.14548)
* [SimCLR](https://arxiv.org/abs/2002.05709) + [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362)
* [SimSiam](https://arxiv.org/abs/2011.10566)
* [Swav](https://arxiv.org/abs/2006.09882)
* [VICReg](https://arxiv.org/abs/2105.04906)

---

## Extra flavor
### Data
* Increased data processing speed by up to 100% using [Nvidia Dali](https://github.com/NVIDIA/DALI)
* Multi-cropping dataloading following [SwAV](https://arxiv.org/abs/2006.09882):
    * **Note**: currently, only SimCLR supports this
* Asymmetric and symmetric augmentations
### Evaluation and logging
* Online linear evaluation via stop-gradient for easier debugging and prototyping (optionally available for the momentum encoder as well)
* Normal offline linear evaluation
* All the perks of PyTorch Lightning (mixed precision, gradient accumulation, clipping, automatic logging and much more)
* Easy-to-extend modular code structure
* Custom model logging with a simpler file organization
* Common metrics and more to come (auto TSNE)

---
## Requirements
* torch
* tqdm
* einops
* wandb
* pytorch-lightning
* lightning-bolts

**Optional**:
* nvidia-dali

**NOTE:** if you are using CUDA 10.X use `nvidia-dali-cuda100` in `requirements.txt`.

---

## Installation
To install the repository with dali support, use:
```
pip3 install -e .[dali]
```

If no dali support is needed, the repository can be installed as:
```
pip3 install .
```

**NOTE:** Soon to be on pip.

---

## Training

For pretraining the encoder, follow one of the many bash files in `bash_files/pretrain/`

After that, for offline linear evaluation, follow the examples on `bash_files/linear`

**NOTE:** Files try to be up-to-date and follow as closely as possible the recommended parameters of each paper, but check them before running.

---

## Results

### CIFAR-10

| Backbone 	| Method       	| Epochs 	| Dali 	| Online linear eval 	| Offline linear eval 	| Checkpoint 	|
|----------	|--------------	|--------	|------	|--------------------	|---------------------	|------------	|
| resnet18 	| Barlow Twins 	|        	|      	|                    	|                     	|            	|
| resnet18 	| BYOL         	|        	|      	|                    	|                     	|            	|
| resnet18 	| MoCo V2+     	|        	|      	|                    	|                     	|            	|
| resnet18 	| NNCLR        	|        	|      	|                    	|                     	|            	|
| resnet18 	| SimCLR       	|        	|      	|                    	|                     	|            	|
| resnet18 	| Simsiam      	|        	|      	|                    	|                     	|            	|
| resnet18 	| VICReg       	|        	|      	|                    	|                     	|            	|


### CIFAR-100

| Backbone 	| Method       	| Epochs 	| Dali 	| Online linear eval 	| Offline linear eval 	| Checkpoint 	|
|----------	|--------------	|--------	|------	|--------------------	|---------------------	|------------	|
| resnet18 	| Barlow Twins 	|        	|      	|                    	|                     	|            	|
| resnet18 	| BYOL         	|        	|      	|                    	|                     	|            	|
| resnet18 	| MoCo V2+     	|        	|      	|                    	|                     	|            	|
| resnet18 	| NNCLR        	|        	|      	|                    	|                     	|            	|
| resnet18 	| SimCLR       	|        	|      	|                    	|                     	|            	|
| resnet18 	| Simsiam      	|        	|      	|                    	|                     	|            	|
| resnet18 	| VICReg       	|        	|      	|                    	|                     	|            	|


### STL-10

| Backbone 	| Method       	| Epochs 	| Dali 	| Online linear eval 	| Offline linear eval 	| Checkpoint 	|
|----------	|--------------	|--------	|------	|--------------------	|---------------------	|------------	|
| resnet18 	| Barlow Twins 	|        	|      	|                    	|                     	|            	|
| resnet18 	| BYOL         	|        	|      	|                    	|                     	|            	|
| resnet18 	| MoCo V2+     	|        	|      	|                    	|                     	|            	|
| resnet18 	| NNCLR        	|        	|      	|                    	|                     	|            	|
| resnet18 	| SimCLR       	|        	|      	|                    	|                     	|            	|
| resnet18 	| Simsiam      	|        	|      	|                    	|                     	|            	|
| resnet18 	| VICReg       	|        	|      	|                    	|                     	|            	|


### Imagenet-100

| Backbone 	| Method       	| Epochs 	| Dali 	| Online linear eval 	| Offline linear eval 	| Checkpoint 	|
|----------	|--------------	|--------	|------	|--------------------	|---------------------	|------------	|
| resnet18 	| Barlow Twins 	| 500    	|      	|                    	|                     	|            	|
| resnet18 	| BYOL         	| 500    	|      	|                    	|                     	|            	|
| resnet18 	| MoCo V2+     	| 500    	|      	|                    	|                     	|            	|
| resnet18 	| NNCLR        	| 500    	|      	|                    	|                     	|            	|
| resnet18 	| SimCLR       	| 500    	|      	|                    	|                     	|            	|
| resnet18 	| Simsiam      	| 500    	|      	|                    	|                     	|            	|
| resnet18 	| VICReg       	| 500    	|      	|                    	|                     	|            	|
### Imagenet
| Method       | Backbone | Epochs |        Dali        | Acc@1 (online) | Acc@1 (offline) | Acc@5 (online) | Acc@5 (offline) | Checkpoint |
|--------------|:--------:|:------:|:------------------:|:--------------:|:---------------:|:--------------:|:---------------:|:----------:|
| Barlow Twins | ResNet50 |        |                    |                |                 |                |                 |            |
| BYOL         | ResNet50 |   100  | :heavy_check_mark: |      65.6      |                 |      86.7      |                 |            |
| MoCo V2+     | ResNet50 |        |                    |                |                 |                |                 |            |
| NNCLR        | ResNet50 |        |                    |                |                 |                |                 |            |
| SimCLR       | ResNet50 |        |                    |                |                 |                |                 |            |
| Simsiam      | ResNet50 |        |                    |                |                 |                |                 |            |
| VICReg       | ResNet50 |        |                    |                |                 |                |                 |            |
<br>

