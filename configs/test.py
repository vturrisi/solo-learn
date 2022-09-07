from omegaconf import OmegaConf

cfg = OmegaConf.load("augmentations/symmetric.yaml")
cfg2 = OmegaConf.load("ddp.yaml")

cfg = OmegaConf.merge(cfg, cfg2)
