*****************
Methods available
*****************

* `Barlow Twins <https://arxiv.org/abs/2103.03230>`_
* `BYOL <https://arxiv.org/abs/2104.14294>`_
* `MoCo-V2 <https://arxiv.org/abs/2003.04297>`_
* `NNCLR <https://arxiv.org/abs/2104.14548>`_
* `SimCLR <https://arxiv.org/abs/2002.05709>`_ + `Supervised Contrastive Learning <https://arxiv.org/abs/2004.11362>`_
* `SimSiam <https://arxiv.org/abs/2011.10566>`_
* `SwAV <https://arxiv.org/abs/2006.09882>`_
* `VICReg <https://arxiv.org/abs/2105.04906>`_
* `W-MSE <https://arxiv.org/abs/2007.06346>`_

************
Extra flavor
************

Data
====

* Increased data processing speed by up to 100% using `Nvidia Dali <https://github.com/NVIDIA/DALI>`_
* Asymmetric and symmetric augmentations

Evaluation and logging
======================


* Online linear evaluation via stop-gradient for easier debugging and prototyping (optionally available for the momentum backbone as well)
* Normal offline linear evaluation
* All the perks of PyTorch Lightning (mixed precision, gradient accumulation, clipping, automatic logging and much more)
* Easy-to-extend modular code structure
* Custom model logging with a simpler file organization
* Automatic feature space visualization with UMAP
* Common metrics and more to come...


Training tricks
===============

* Multi-cropping dataloading following `SwAV <https://arxiv.org/abs/2006.09882>`_:
    * **Note**: currently, only SimCLR supports this
* Exclude batchnorm and biases from LARS
* No LR scheduler for the projection head in SimSiam
