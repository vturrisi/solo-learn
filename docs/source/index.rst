.. solo-learn documentation master file, created by
   sphinx-quickstart on Thu Jul  8 14:57:18 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

####################
solo-learn
####################

A library of self-supervised methods for unsupervised visual representation learning powered by PyTorch Lightning.
We aim at providing SOTA self-supervised methods in a comparable environment while, at the same time, implementing training tricks.
While the library is self contained, it is possible to use the models outside of solo-learn.

======================================

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   start/install
   start/available

.. toctree::
   :maxdepth: 2
   :caption: Documentation

   solo/args
   solo/losses
   solo/methods/base
   solo/methods/dali
   solo/methods/linear
   solo/methods/barlow
   solo/methods/byol
   solo/methods/dino
   solo/methods/mocov2plus
   solo/methods/nnclr
   solo/methods/simclr
   solo/methods/simsiam
   solo/methods/swav
   solo/methods/vicreg
   solo/methods/wmse
   solo/utils

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
