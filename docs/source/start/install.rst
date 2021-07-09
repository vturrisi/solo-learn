Installation
************

To install the repository with dali support, use:

.. code-block:: python

   pip instal .[dali]

If no dali support is needed, the repository can be installed as:


.. code-block:: python

   pip instal .



**NOTE:** If you want to modify the library, install it in dev mode with `-e`.

**NOTE 2:** Soon to be on pip.

**Requirements:**

.. code-block:: python

    torch
    tqdm
    einops
    wandb
    pytorch-lightning
    lightning-bolts

    # Optional
    nvidia-dali

**NOTE:** if you are using CUDA 10.X use `nvidia-dali-cuda100` in `requirements.txt`.
