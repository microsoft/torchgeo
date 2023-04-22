import os

from hydra.compose import OmegaConf  # type: ignore[attr-defined]
from hydra.utils import instantiate


def test_recursive_config() -> None:
    conf = OmegaConf.load(os.path.join("tests", "conf", "trainer.yaml"))
    instantiate(conf).trainer
