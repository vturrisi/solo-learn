solo.utils
==========


Checkpointer
------------

__init__
~~~~~~~~
.. automethod:: solo.utils.checkpointer.Checkpointer.__init__
   :noindex:

add_checkpointer_args
~~~~~~~~~~~~~~~~~~~~~
.. automethod:: solo.utils.checkpointer.Checkpointer.add_checkpointer_args
   :noindex:

initial_setup
~~~~~~~~~~~~~
.. automethod:: solo.utils.checkpointer.Checkpointer.initial_setup
   :noindex:

save_args
~~~~~~~~~
.. automethod:: solo.utils.checkpointer.Checkpointer.save_args
   :noindex:

save
~~~~
.. automethod:: solo.utils.checkpointer.Checkpointer.save
   :noindex:

on_train_start
~~~~~~~~~~~~~~
.. automethod:: solo.utils.checkpointer.Checkpointer.on_train_start
   :noindex:

on_validation_end
~~~~~~~~~~~~~~~~~
.. automethod:: solo.utils.checkpointer.Checkpointer.on_validation_end
   :noindex:



Classification dataloaders
--------------------------

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
-------------

GaussianBlur
~~~~~~~~~~~~
.. autoclass:: solo.utils.contrastive_dataloader.GaussianBlur
   :noindex:

Solarization
~~~~~~~~~~~~
.. autoclass:: solo.utils.contrastive_dataloader.Solarization
   :noindex:

NCropAugmentation
~~~~~~~~~~~~~~~~~
.. autoclass:: solo.utils.contrastive_dataloader.NCropAugmentation
   :noindex:



Transformation Pipelines
------------------------

BaseTransform
~~~~~~~~~~~~~
.. automethod:: solo.utils.contrastive_dataloader.BaseTransform
   :noindex:

CifarTransform
~~~~~~~~~~~~~~
.. automethod:: solo.utils.contrastive_dataloader.CifarTransform.__init__
   :noindex:

STLTransform
~~~~~~~~~~~~
.. automethod:: solo.utils.contrastive_dataloader.STLTransform.__init__
   :noindex:

ImagenetTransform
~~~~~~~~~~~~~~~~~
.. automethod:: solo.utils.contrastive_dataloader.ImagenetTransform.__init__
   :noindex:

MulticropAugmentation
~~~~~~~~~~~~~~~~~~~~~
.. automethod:: solo.utils.contrastive_dataloader.MulticropAugmentation.__init__
   :noindex:

MulticropCifarTransform
~~~~~~~~~~~~~~~~~~~~~~~
.. automethod:: solo.utils.contrastive_dataloader.MulticropCifarTransform.__init__
   :noindex:

MulticropSTLTransform
~~~~~~~~~~~~~~~~~~~~~
.. automethod:: solo.utils.contrastive_dataloader.MulticropSTLTransform.__init__
   :noindex:

MulticropImagenetTransform
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automethod:: solo.utils.contrastive_dataloader.MulticropImagenetTransform.__init__
   :noindex:


Contrastive dataloaders
-----------------------

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
----------------

.. automodule:: solo.utils.dali_dataloader
   :members:
   :undoc-members:
   :show-inheritance:

Gather layer
------------

.. automodule:: solo.utils.gather_layer
   :members:
   :undoc-members:
   :show-inheritance:

LARS
----

.. automodule:: solo.utils.lars
   :members:
   :undoc-members:
   :show-inheritance:

Metrics
-------

.. automodule:: solo.utils.metrics
   :members:
   :undoc-members:
   :show-inheritance:

Momentum module
---------------

.. automodule:: solo.utils.momentum
   :members:
   :undoc-members:
   :show-inheritance:

Sinkhorn-Knopp
--------------

.. automodule:: solo.utils.sinkhorn_knopp
   :members:
   :undoc-members:
   :show-inheritance:

Truncate normal
---------------

.. automodule:: solo.utils.trunc_normal
   :members:
   :undoc-members:
   :show-inheritance:

Whitening
---------

.. automodule:: solo.utils.whitening
   :members:
   :undoc-members:
   :show-inheritance:
