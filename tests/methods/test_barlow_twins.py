from solo.methods import BarlowTwins
from .utils import gen_base_kwargs, DATA_KWARGS, gen_batch


def test_barlow():
    barlow_kwargs = {"proj_hidden_dim": 2048, "output_dim": 2048, "lamb": 5e-3, "scale_loss": 0.025}

    BASE_KWARGS = gen_base_kwargs(cifar=False)
    kwargs = {**BASE_KWARGS, **DATA_KWARGS, **barlow_kwargs}
    bt = BarlowTwins(**kwargs)

    batch, batch_idx = gen_batch(10, "imagenet100")
    loss = bt.training_step(batch, batch_idx)

    assert loss != 0

    BASE_KWARGS = gen_base_kwargs(cifar=True)
    kwargs = {**BASE_KWARGS, **DATA_KWARGS, **barlow_kwargs}
    bt = BarlowTwins(**kwargs)

    batch, batch_idx = gen_batch(10, "cifar10")
    loss = bt.training_step(batch, batch_idx)

    assert loss != 0
