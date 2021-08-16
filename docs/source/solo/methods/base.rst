BaseModel
=========


.. automethod:: solo.methods.base.BaseModel.__init__
   :noindex:

add_model_specific_args
~~~~~~~~~~~~~~~~~~~~~~~
.. automethod:: solo.methods.base.BaseModel.add_model_specific_args
   :noindex:

learnable_params
~~~~~~~~~~~~~~~~
.. autoattribute:: solo.methods.base.BaseModel.learnable_params
   :noindex:

configure_optimizers
~~~~~~~~~~~~~~~~~~~~
.. automethod:: solo.methods.base.BaseModel.configure_optimizers
   :noindex:

forward
~~~~~~~
.. automethod:: solo.methods.base.BaseModel.forward
   :noindex:

base_forward
~~~~~~~~~~~~~
.. automethod:: solo.methods.base.BaseModel.base_forward
   :noindex:

_shared_step
~~~~~~~~~~~~
.. automethod:: solo.methods.base.BaseModel._shared_step
   :noindex:

training_step
~~~~~~~~~~~~~
.. automethod:: solo.methods.base.BaseModel.training_step
   :noindex:

validation_step
~~~~~~~~~~~~~~~
.. automethod:: solo.methods.base.BaseModel.validation_step
   :noindex:

validation_epoch_end
~~~~~~~~~~~~~~~~~~~~
.. automethod:: solo.methods.base.BaseModel.validation_epoch_end
   :noindex:



BaseMomentumModel
-----------------


.. automethod:: solo.methods.base.BaseMomentumModel.__init__
   :noindex:

learnable_params
~~~~~~~~~~~~~~~~
.. autoattribute:: solo.methods.base.BaseMomentumModel.learnable_params
   :noindex:

momentum_pairs
~~~~~~~~~~~~~~
.. autoattribute:: solo.methods.base.BaseMomentumModel.momentum_pairs
   :noindex:

add_model_specific_args
~~~~~~~~~~~~~~~~~~~~~~~
.. automethod:: solo.methods.base.BaseMomentumModel.add_model_specific_args
   :noindex:

on_train_start
~~~~~~~~~~~~~~
.. automethod:: solo.methods.base.BaseMomentumModel.on_train_start
   :noindex:

base_momentum_forward
~~~~~~~~~~~~~~
.. automethod:: solo.methods.base.BaseMomentumModel.base_momentum_forward
   :noindex:

_shared_step_momentum
~~~~~~~~~~~~~~~~~~~~~
.. automethod:: solo.methods.base.BaseMomentumModel._shared_step_momentum
   :noindex:

training_step
~~~~~~~~~~~~~
.. automethod:: solo.methods.base.BaseModel.training_step
   :noindex:

on_train_batch_end
~~~~~~~~~~~~~~~~~~
.. automethod:: solo.methods.base.BaseMomentumModel.on_train_batch_end
   :noindex:

validation_step
~~~~~~~~~~~~~~~
.. automethod:: solo.methods.base.BaseModel.validation_step
   :noindex:

validation_epoch_end
~~~~~~~~~~~~~~~~~~~~
.. automethod:: solo.methods.base.BaseModel.validation_epoch_end
   :noindex:
