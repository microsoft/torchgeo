import os

from hydra.utils import instantiate
from omegaconf import OmegaConf


def test_recursive_config() -> None:
    conf = OmegaConf.load(os.path.join("tests", "conf", "trainer.yaml"))
    instantiate(conf).trainer
