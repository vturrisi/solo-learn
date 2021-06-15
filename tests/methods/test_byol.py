from solo.methods import BYOL
from .utils import gen_base_kwargs, DATA_KWARGS, gen_batch


def test_byol():
    method_kwargs = {"output_dim": 256, "proj_hidden_dim": 2048, "pred_hidden_dim": 2048}

    BASE_KWARGS = gen_base_kwargs(cifar=False, momentum=True)
    kwargs = {**BASE_KWARGS, **DATA_KWARGS, **method_kwargs}
    model = BYOL(**kwargs)

    batch, batch_idx = gen_batch(BASE_KWARGS["batch_size"], BASE_KWARGS["n_classes"], "imagenet100")
    loss = model.training_step(batch, batch_idx)

    assert loss != 0

    BASE_KWARGS = gen_base_kwargs(cifar=True, momentum=True)
    kwargs = {**BASE_KWARGS, **DATA_KWARGS, **method_kwargs}
    model = BYOL(**kwargs)

    batch, batch_idx = gen_batch(BASE_KWARGS["batch_size"], BASE_KWARGS["n_classes"], "cifar10")
    loss = model.training_step(batch, batch_idx)

    assert loss != 0
