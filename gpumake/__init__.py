"""gpumake: a make-style pipeline runner for shared GPU servers.

Inputs are hashed into a data revision (``runs/<revision>``); code is grouped into units whose
outputs are superseded and rebuilt only when their files change; jobs run side by side on idle
GPUs and resume after an interruption. ``gpumake.remote`` drives it from a workstation over SSH.
"""
from .spec import Job, Unit

__all__ = ["Job", "Unit"]
__version__ = "0.1.0"
