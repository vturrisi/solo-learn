[![unit-tests](https://github.com/vturrisi/solo-learn/actions/workflows/tests.yml/badge.svg)](https://github.com/vturrisi/solo-learn/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/vturrisi/solo-learn/branch/main/graph/badge.svg?token=WLU9UU17XZ)](https://codecov.io/gh/vturrisi/solo-learn)

# solo-learn

A library of self-supervised methods for unsupervised visual representation learning powered by PyTorch Lightning.
We aim at providing SOTA self-supervised methods in a comparable environment while, at the same time, implementing training tricks.
While the library is self contained, it is possible to use the models outside of solo-learn.

---

## Methods available:
* [Barlow Twins](https://arxiv.org/abs/2103.03230)
* [BYOL](https://arxiv.org/abs/2006.07733)
* [DINO](https://arxiv.org/abs/2104.14294)
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
* Asymmetric and symmetric augmentations
### Evaluation and logging
* Online linear evaluation via stop-gradient for easier debugging and prototyping (optionally available for the momentum encoder as well)
* Normal offline linear evaluation
* All the perks of PyTorch Lightning (mixed precision, gradient accumulation, clipping, automatic logging and much more)
* Easy-to-extend modular code structure
* Custom model logging with a simpler file organization
* Common metrics and more to come (auto TSNE)
### Training tricks
* Multi-cropping dataloading following [SwAV](https://arxiv.org/abs/2006.09882):
    * **Note**: currently, only SimCLR supports this
* Exclude batchnorm and biases from LARS
* No LR scheduler for the projection head in SimSiam
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
pip3 install .[dali]
```

If no dali support is needed, the repository can be installed as:
```
pip3 install .
```

**NOTE:** If you want to modify the library, install it in dev mode with `-e`.

**NOTE 2:** Soon to be on pip.

---

## Training

For pretraining the encoder, follow one of the many bash files in `bash_files/pretrain/`

After that, for offline linear evaluation, follow the examples on `bash_files/linear`

**NOTE:** Files try to be up-to-date and follow as closely as possible the recommended parameters of each paper, but check them before running.

---

## Results

### CIFAR-10

| Method       | Backbone | Epochs | Dali | Acc@1 (online) | Acc@1 (offline) | Acc@5 (online) | Acc@5 (offline) | Checkpoint |
|--------------|:--------:|:------:|:----:|:--------------:|:---------------:|:--------------:|:---------------:|:----------:|
| Barlow Twins | ResNet18 |  1000  |  :x: |      92.10     |                 |     99.73      |                 |            |
| BYOL         | ResNet18 |  1000  |  :x: |      92.58     |                 |     99.79      |                 |            |
| DINO         | ResNet18 |  1000  |  :x: |      89.52     |                 |     99.71      |                 |            |
| MoCo V2+     | ResNet18 |  1000  |  :x: |                |                 |                |                 |            |
| NNCLR        | ResNet18 |  1000  |  :x: |      91.88     |                 |     99.78      |                 |            |
| SimCLR       | ResNet18 |  1000  |  :x: |      90.74     |                 |     99.75      |                 |            |
| Simsiam      | ResNet18 |  1000  |  :x: |                |                 |                |                 |            |
| SwAV         | ResNet18 |  1000  |  :x: |      89.17     |                 |     99.68      |                 |            |
| VICReg       | ResNet18 |  1000  |  :x: |      92.07     |                 |     99.74      |                 |            |

### CIFAR-100

| Method       | Backbone | Epochs | Dali | Acc@1 (online) | Acc@1 (offline) | Acc@5 (online) | Acc@5 (offline) | Checkpoint |
|--------------|:--------:|:------:|:----:|:--------------:|:---------------:|:--------------:|:---------------:|:----------:|
| Barlow Twins | ResNet18 |  1000  |  :x: |      70.90     |                 |     91.91      |                 |            |
| BYOL         | ResNet18 |  1000  |  :x: |      70.46     |                 |     91.96      |                 |            |
| DINO         | ResNet18 |  1000  |  :x: |      66.76     |                 |     90.34      |                 |            |
| MoCo V2+     | ResNet18 |  1000  |  :x: |                |                 |                |                 |            |
| NNCLR        | ResNet18 |  1000  |  :x: |      69.62     |                 |     91.52      |                 |            |
| SimCLR       | ResNet18 |  1000  |  :x: |      65.78     |                 |     89.04      |                 |            |
| Simsiam      | ResNet18 |  1000  |  :x: |                |                 |                |                 |            |
| SwAV         | ResNet18 |  1000  |  :x: |      64.88     |                 |     88.78      |                 |            |
| VICReg       | ResNet18 |  1000  |  :x: |      68.54     |                 |     90.83      |                 |            |

### ImageNet-100


| Method       | Backbone | Epochs |        Dali        | Acc@1 (online) | Acc@1 (offline) | Acc@5 (online) | Acc@5 (offline) | Checkpoint |
|--------------|:--------:|:------:|:------------------:|:--------------:|:---------------:|:--------------:|:---------------:|:----------:|
| Barlow Twins | ResNet50 |   500  | :heavy_check_mark: |      79.80     |                 |      95.28     |                 |            |
| BYOL         | ResNet50 |   500  | :heavy_check_mark: |      78.88     |                 |      94.52     |                 |            |
| DINO         | ResNet50 |   500  | :heavy_check_mark: |                |                 |                |                 |            |
| MoCo V2+     | ResNet50 |   500  | :heavy_check_mark: |                |                 |                |                 |            |
| NNCLR        | ResNet50 |   500  | :heavy_check_mark: |                |                 |                |                 |            |
| SimCLR       | ResNet50 |   500  | :heavy_check_mark: |                |                 |                |                 |            |
| Simsiam      | ResNet50 |   500  | :heavy_check_mark: |                |                 |                |                 |            |
| SwAV         | ResNet50 |   500  | :heavy_check_mark: |                |                 |                |                 |            |
| VICReg       | ResNet50 |   500  | :heavy_check_mark: |                |                 |                |                 |            |

### ImageNet
| Method       | Backbone | Epochs |        Dali        | Acc@1 (online) | Acc@1 (offline) | Acc@5 (online) | Acc@5 (offline) | Checkpoint |
|--------------|:--------:|:------:|:------------------:|:--------------:|:---------------:|:--------------:|:---------------:|:----------:|
| Barlow Twins | ResNet50 |   100  | :heavy_check_mark: |                |                 |                |                 |            |
| BYOL         | ResNet50 |   100  | :heavy_check_mark: |      65.6      |      67.07      |      86.7      |      87.81      |            |
| DINO         | ResNet50 |   100  | :heavy_check_mark: |                |                 |                |                 |            |
| MoCo V2+     | ResNet50 |   100  | :heavy_check_mark: |                |                 |                |                 |            |
| NNCLR        | ResNet50 |   100  | :heavy_check_mark: |                |                 |                |                 |            |
| SimCLR       | ResNet50 |   100  | :heavy_check_mark: |                |                 |                |                 |            |
| Simsiam      | ResNet50 |   100  | :heavy_check_mark: |                |                 |                |                 |            |
| SwAV         | ResNet50 |   500  | :heavy_check_mark: |                |                 |                |                 |            |
| VICReg       | ResNet50 |   100  | :heavy_check_mark: |                |                 |                |                 |            |

## Training efficiency
Our standardized implementation enables a fair comparison of training efficiency. Here we report the training time and memory usage on ImageNet-100 using ResNet50 and running on 2 Quadro RTX 6000.

| Method       | Dali | Parameters | Learnable parameters | Time for 1 epoch | GPU memory |
|--------------|------|------------|----------------------|------------------|------------|
| Barlow Twins |      |            |                      |                  |            |
| BYOL         |      |            |                      |                  |            |
| DINO         |      |            |                      |                  |            |
| MoCo V2+     |      |            |                      |                  |            |
| NNCLR        |      |            |                      |                  |            |
| SimCLR       |      |            |                      |                  |            |
| Simsiam      |      |            |                      |                  |            |
| SwAV         |      |            |                      |                  |            |
| VICReg       |      |            |                      |                  |            |
<br>

