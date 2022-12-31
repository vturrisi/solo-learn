Classification dataloaders
==========================

prepare_transforms
~~~~~~~~~~~~~~~~~~
.. autofunction:: solo.data.classification_dataloader.prepare_transforms
   :noindex:

prepare_datasets
~~~~~~~~~~~~~~~~
.. autofunction:: solo.data.classification_dataloader.prepare_datasets
   :noindex:

prepare_dataloaders
~~~~~~~~~~~~~~~~~~~
.. autofunction:: solo.data.classification_dataloader.prepare_dataloaders
   :noindex:

prepare_data
~~~~~~~~~~~~
.. autofunction:: solo.data.classification_dataloader.prepare_data
   :noindex:


Augmentations
=============

GaussianBlur
~~~~~~~~~~~~
.. autoclass:: solo.data.pretrain_dataloader.GaussianBlur
   :noindex:

Solarization
~~~~~~~~~~~~
.. autoclass:: solo.data.pretrain_dataloader.Solarization
   :noindex:

NCropAugmentation
~~~~~~~~~~~~~~~~~
.. autoclass:: solo.data.pretrain_dataloader.NCropAugmentation
   :noindex:



Transformation Pipelines
========================

BaseTransform
~~~~~~~~~~~~~
.. automethod:: solo.data.pretrain_dataloader.BaseTransform
   :noindex:

CifarTransform
~~~~~~~~~~~~~~
.. automethod:: solo.data.pretrain_dataloader.CifarTransform.__init__
   :noindex:

STLTransform
~~~~~~~~~~~~
.. automethod:: solo.data.pretrain_dataloader.STLTransform.__init__
   :noindex:

ImagenetTransform
~~~~~~~~~~~~~~~~~
.. automethod:: solo.data.pretrain_dataloader.ImagenetTransform.__init__
   :noindex:

CustomTransform
~~~~~~~~~~~~~~~
.. automethod:: solo.data.pretrain_dataloader.CustomTransform.__init__
   :noindex:



Pretrain dataloader
===================

dataset_with_index
~~~~~~~~~~~~~~~~~~
.. autofunction:: solo.data.pretrain_dataloader.dataset_with_index
   :noindex:

prepare_transform
~~~~~~~~~~~~~~~~~
.. autofunction:: solo.data.pretrain_dataloader.prepare_transform
   :noindex:

prepare_n_crop_transform
~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: solo.data.pretrain_dataloader.prepare_n_crop_transform
   :noindex:

prepare_datasets
~~~~~~~~~~~~~~~~
.. autofunction:: solo.data.pretrain_dataloader.prepare_datasets
   :noindex:

prepare_dataloader
~~~~~~~~~~~~~~~~~~~
.. autofunction:: solo.data.pretrain_dataloader.prepare_dataloader
   :noindex:


DALI dataloaders
================

.. automodule:: solo.data.dali_dataloader
   :members:
