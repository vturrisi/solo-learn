from solo.methods import VICReg
from .utils import gen_base_kwargs, DATA_KWARGS, gen_batch


def test_vicreg():
    method_kwargs = {
        "proj_hidden_dim": 2048,
        "output_dim": 2048,
        "sim_loss_weight": 25.0,
        "var_loss_weight": 25.0,
        "cov_loss_weight": 1.0,
    }

    BASE_KWARGS = gen_base_kwargs(cifar=False)
    kwargs = {**BASE_KWARGS, **DATA_KWARGS, **method_kwargs}
    model = VICReg(**kwargs)

    batch, batch_idx = gen_batch(BASE_KWARGS["batch_size"], BASE_KWARGS["n_classes"], "imagenet100")
    loss = model.training_step(batch, batch_idx)

    assert loss != 0

    BASE_KWARGS = gen_base_kwargs(cifar=True)
    kwargs = {**BASE_KWARGS, **DATA_KWARGS, **method_kwargs}
    model = VICReg(**kwargs)

    batch, batch_idx = gen_batch(BASE_KWARGS["batch_size"], BASE_KWARGS["n_classes"], "cifar10")
    loss = model.training_step(batch, batch_idx)

    assert loss != 0
