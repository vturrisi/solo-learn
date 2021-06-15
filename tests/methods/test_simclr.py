from solo.methods import SimCLR
from .utils import gen_base_kwargs, DATA_KWARGS, gen_batch


def test_simclr():
    method_kwargs = {
        "output_dim": 256,
        "proj_hidden_dim": 2048,
        "temperature": 0.2,
        "supervised": False,
    }

    BASE_KWARGS = gen_base_kwargs(cifar=False)
    kwargs = {**BASE_KWARGS, **DATA_KWARGS, **method_kwargs}
    model = SimCLR(**kwargs)

    batch, batch_idx = gen_batch(BASE_KWARGS["batch_size"], BASE_KWARGS["n_classes"], "imagenet100")
    loss = model.training_step(batch, batch_idx)

    assert loss != 0

    BASE_KWARGS = gen_base_kwargs(cifar=True)
    kwargs = {**BASE_KWARGS, **DATA_KWARGS, **method_kwargs}
    model = SimCLR(**kwargs)

    batch, batch_idx = gen_batch(BASE_KWARGS["batch_size"], BASE_KWARGS["n_classes"], "cifar10")
    loss = model.training_step(batch, batch_idx)

    assert loss != 0

    method_kwargs = {
        "output_dim": 256,
        "proj_hidden_dim": 2048,
        "temperature": 0.2,
        "supervised": True,
    }

    BASE_KWARGS = gen_base_kwargs(cifar=False)
    kwargs = {**BASE_KWARGS, **DATA_KWARGS, **method_kwargs}
    model = SimCLR(**kwargs)

    batch, batch_idx = gen_batch(BASE_KWARGS["batch_size"], BASE_KWARGS["n_classes"], "imagenet100")
    loss = model.training_step(batch, batch_idx)

    assert loss != 0
