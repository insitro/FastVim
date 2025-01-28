__version__ = "1.1.1"

from mamba_ssm.modules.mamba_simple import Mamba as Mamba
from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn as mamba_inner_fn
from mamba_ssm.ops.selective_scan_interface import (
    selective_scan_fn as selective_scan_fn,
)
