from .marida import MARIDA, MARIDADataModule
from .mados import MADOS, MADOSDataModule
from .magicbathynet import MagicBathyNet, MagicBathyNetDataModule
from .ships2ais import ShipS2AIS
from .swed import SWED, SWEDDataModule
from . import transforms

__all__ = [
    "MARIDA",
    "MARIDADataModule",
    "MADOS",
    "MADOSDataModule",
    "MagicBathyNet",
    "MagicBathyNetDataModule",
    "ShipS2AIS",
    "SWED",
    "SWEDDataModule",
    "transforms",
]
