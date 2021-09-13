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

CustomTransform
~~~~~~~~~~~~~~~
.. automethod:: solo.utils.pretrain_dataloader.CustomTransform.__init__
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

MulticropCustomTransform
~~~~~~~~~~~~~~~~~~~~~~~~
.. automethod:: solo.utils.pretrain_dataloader.MulticropCustomTransform.__init__
   :noindex:



Pretrain dataloader
===================

dataset_with_index
~~~~~~~~~~~~~~~~~~
.. autofunction:: solo.utils.pretrain_dataloader.dataset_with_index
   :noindex:

prepare_transform
~~~~~~~~~~~~~~~~~
.. autofunction:: solo.utils.pretrain_dataloader.prepare_transform
   :noindex:

prepare_n_crop_transform
~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: solo.utils.pretrain_dataloader.prepare_n_crop_transform
   :noindex:

prepare_multicrop_transform
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: solo.utils.pretrain_dataloader.prepare_multicrop_transform
   :noindex:

prepare_datasets
~~~~~~~~~~~~~~~~
.. autofunction:: solo.utils.pretrain_dataloader.prepare_datasets
   :noindex:

prepare_dataloader
~~~~~~~~~~~~~~~~~~~
.. autofunction:: solo.utils.pretrain_dataloader.prepare_dataloader
   :noindex:


DALI dataloaders
================

.. automodule:: solo.utils.dali_dataloader
   :members: