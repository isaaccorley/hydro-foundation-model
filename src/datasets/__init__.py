from .earth_surface_water import EarthSurfaceWater, EarthSurfaceWaterDataModule
from .marida import MARIDA, MARIDADataModule
from .mados import MADOS, MADOSDataModule
from .magicbathynet import MagicBathyNet  # , MagicBathyNetDataModule
from .ships2ais import ShipS2AIS, ShipS2AISDataModule
from .swed import SWED, SWEDDataModule
from . import transforms

__all__ = [
    "EarthSurfaceWater",
    "EarthSurfaceWaterDataModule",
    "MARIDA",
    "MARIDADataModule",
    "MADOS",
    "MADOSDataModule",
    "MagicBathyNet",
    # "MagicBathyNetDataModule",
    "ShipS2AIS",
    "ShipS2AISDataModule",
    "SWED",
    "SWEDDataModule",
    "transforms",
]
