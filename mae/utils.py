import os
from pathlib import Path
from typing import List, Optional, TypeVar, Union

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, basecontainer

T = TypeVar("T")


def load_config(
    config_path: Union[str, Path],
    config_name: str,
    overrides: Optional[List[str]] = None,
) -> DictConfig:
    """Load a Hydra configuration given a path and file name
    Parameters
    ----------
    config_path: str
        Path to the Hydra config folder.
    config_name: str
        Name of the Hydra config file.
    overrides: Optional[List[str]]
        Overrides for values defined in the config.
        See https://hydra.cc/docs/advanced/override_grammar/basic/ for syntax
    """
    # Hydra expects the config_path to be relative to the _calling file_, i.e. this file.
    # Since we're taking config_path as a command line argument we want to treat it as
    # relative to the current working directory. We do the conversion here.
    desired_path = Path.cwd() / config_path
    print(desired_path)
    relative_path = os.path.relpath(desired_path, Path(__file__).parent)
    print(relative_path)
    with hydra.initialize(version_base=None, config_path=relative_path):
        config = hydra.compose(config_name=config_name, overrides=overrides)
    return config


def _maybe_instantiate(config: Union[T, DictConfig]) -> T:
    if isinstance(config, basecontainer.BaseContainer):
        return instantiate(config)
    else:
        return config
