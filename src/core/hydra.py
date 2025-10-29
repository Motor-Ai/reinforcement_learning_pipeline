from hydra.utils import get_class
from hydra.utils import instantiate
from omegaconf import OmegaConf


def instantiate_frozen(cfg):
    """
    Like hydra.utils.instantiate, but:
    - passes config as a single immutable argument
    - forbids mutation inside
    """
    if "_target_" not in cfg:
        raise ValueError("Missing _target_ in config")

    # Import and get target class/function
    target = get_class(cfg._target_)

    # Remove _target_ key
    cfg_copy = OmegaConf.to_container(cfg, resolve=True)
    cfg_copy.pop("_target_", None)

    # Convert to read-only DictConfig
    frozen_cfg = OmegaConf.create(cfg_copy)
    OmegaConf.set_readonly(frozen_cfg, True)

    # Instantiate object with config argument
    return target(config=frozen_cfg)
