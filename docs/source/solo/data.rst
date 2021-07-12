Classification dataloaders
==========================

prepare_transforms
~~~~~~~~~~~~~~~~~~
.. autofunction:: solo.utils.classification_dataloader.prepare_transforms
   :noindex:

prepare_datasets
~~~~~~~~~~~~~~~~
.. autofunction:: solo.utils.classification_dataloader.prepare_datasets
   :noindex:

prepare_dataloaders
~~~~~~~~~~~~~~~~~~~
.. autofunction:: solo.utils.classification_dataloader.prepare_dataloaders
   :noindex:

prepare_data
~~~~~~~~~~~~
.. autofunction:: solo.utils.classification_dataloader.prepare_data
   :noindex:


Augmentations
=============

GaussianBlur
~~~~~~~~~~~~
.. autoclass:: solo.utils.pretrain_dataloader.GaussianBlur
   :noindex:

Solarization
~~~~~~~~~~~~
.. autoclass:: solo.utils.pretrain_dataloader.Solarization
   :noindex:

NCropAugmentation
~~~~~~~~~~~~~~~~~
.. autoclass:: solo.utils.pretrain_dataloader.NCropAugmentation
   :noindex:



Transformation Pipelines
========================

BaseTransform
~~~~~~~~~~~~~
.. automethod:: solo.utils.pretrain_dataloader.BaseTransform
   :noindex:

CifarTransform
~~~~~~~~~~~~~~
.. automethod:: solo.utils.pretrain_dataloader.CifarTransform.__init__
   :noindex:

STLTransform
~~~~~~~~~~~~
.. automethod:: solo.utils.pretrain_dataloader.STLTransform.__init__
   :noindex:

ImagenetTransform
~~~~~~~~~~~~~~~~~
.. automethod:: solo.utils.pretrain_dataloader.ImagenetTransform.__init__
   :noindex:

MulticropAugmentation
~~~~~~~~~~~~~~~~~~~~~
.. automethod:: solo.utils.pretrain_dataloader.MulticropAugmentation.__init__
   :noindex:

MulticropCifarTransform
~~~~~~~~~~~~~~~~~~~~~~~
.. automethod:: solo.utils.pretrain_dataloader.MulticropCifarTransform.__init__
   :noindex:

MulticropSTLTransform
~~~~~~~~~~~~~~~~~~~~~
.. automethod:: solo.utils.pretrain_dataloader.MulticropSTLTransform.__init__
   :noindex:

MulticropImagenetTransform
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automethod:: solo.utils.pretrain_dataloader.MulticropImagenetTransform.__init__
   :noindex:


Contrastive dataloaders
=======================

dataset_with_index
~~~~~~~~~~~~~~~~~~
.. autofunction:: solo.utils.classification_dataloader.prepare_data
   :noindex:

prepare_transform
~~~~~~~~~~~~~~~~~
.. autofunction:: solo.utils.classification_dataloader.prepare_data
   :noindex:

prepare_n_crop_transform
~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: solo.utils.classification_dataloader.prepare_data
   :noindex:

prepare_multicrop_transform
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: solo.utils.classification_dataloader.prepare_data
   :noindex:

prepare_datasets
~~~~~~~~~~~~~~~~
.. autofunction:: solo.utils.classification_dataloader.prepare_data
   :noindex:

prepare_dataloaders
~~~~~~~~~~~~~~~~~~~
.. autofunction:: solo.utils.classification_dataloader.prepare_data
   :noindex:


DALI dataloaders
================

.. automodule:: solo.utils.dali_dataloader
   :members: