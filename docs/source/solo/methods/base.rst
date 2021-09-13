BaseMethod
==========


.. automethod:: solo.methods.base.BaseMethod.__init__
   :noindex:

add_model_specific_args
~~~~~~~~~~~~~~~~~~~~~~~
.. automethod:: solo.methods.base.BaseMethod.add_model_specific_args
   :noindex:

learnable_params
~~~~~~~~~~~~~~~~
.. autoattribute:: solo.methods.base.BaseMethod.learnable_params
   :noindex:

configure_optimizers
~~~~~~~~~~~~~~~~~~~~
.. automethod:: solo.methods.base.BaseMethod.configure_optimizers
   :noindex:

forward
~~~~~~~
.. automethod:: solo.methods.base.BaseMethod.forward
   :noindex:

base_forward
~~~~~~~~~~~~~
.. automethod:: solo.methods.base.BaseMethod.base_forward
   :noindex:

_shared_step
~~~~~~~~~~~~
.. automethod:: solo.methods.base.BaseMethod._shared_step
   :noindex:

training_step
~~~~~~~~~~~~~
.. automethod:: solo.methods.base.BaseMethod.training_step
   :noindex:

validation_step
~~~~~~~~~~~~~~~
.. automethod:: solo.methods.base.BaseMethod.validation_step
   :noindex:

validation_epoch_end
~~~~~~~~~~~~~~~~~~~~
.. automethod:: solo.methods.base.BaseMethod.validation_epoch_end
   :noindex:



BaseMomentumMethod
==================


.. automethod:: solo.methods.base.BaseMomentumMethod.__init__
   :noindex:

learnable_params
~~~~~~~~~~~~~~~~
.. autoattribute:: solo.methods.base.BaseMomentumMethod.learnable_params
   :noindex:

momentum_pairs
~~~~~~~~~~~~~~
.. autoattribute:: solo.methods.base.BaseMomentumMethod.momentum_pairs
   :noindex:

add_model_specific_args
~~~~~~~~~~~~~~~~~~~~~~~
.. automethod:: solo.methods.base.BaseMomentumMethod.add_model_specific_args
   :noindex:

on_train_start
~~~~~~~~~~~~~~
.. automethod:: solo.methods.base.BaseMomentumMethod.on_train_start
   :noindex:

base_momentum_forward
~~~~~~~~~~~~~~~~~~~~~
.. automethod:: solo.methods.base.BaseMomentumMethod.base_momentum_forward
   :noindex:

_shared_step_momentum
~~~~~~~~~~~~~~~~~~~~~
.. automethod:: solo.methods.base.BaseMomentumMethod._shared_step_momentum
   :noindex:

training_step
~~~~~~~~~~~~~
.. automethod:: solo.methods.base.BaseMethod.training_step
   :noindex:

on_train_batch_end
~~~~~~~~~~~~~~~~~~
.. automethod:: solo.methods.base.BaseMomentumMethod.on_train_batch_end
   :noindex:

validation_step
~~~~~~~~~~~~~~~
.. automethod:: solo.methods.base.BaseMethod.validation_step
   :noindex:

validation_epoch_end
~~~~~~~~~~~~~~~~~~~~
.. automethod:: solo.methods.base.BaseMethod.validation_epoch_end
   :noindex:
