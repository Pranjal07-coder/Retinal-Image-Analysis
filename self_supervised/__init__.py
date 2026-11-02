"""Self-supervised learning modules"""

from .simclr import SimCLR, SimCLRAugmentation
from .byol import BYOL, BYOLAugmentation

__all__ = [
    'SimCLR',
    'SimCLRAugmentation',
    'BYOL',
    'BYOLAugmentation'
]
