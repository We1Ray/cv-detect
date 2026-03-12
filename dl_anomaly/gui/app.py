"""Backward-compatibility shim.

The original ``MainApp`` has been replaced by :class:`HalconApp` in
``gui.halcon_app``.  This module re-exports it under the old name so that
any code that does ``from dl_anomaly.gui.app import MainApp`` continues
to work.
"""

from dl_anomaly.gui.halcon_app import HalconApp as MainApp

__all__ = ["MainApp"]
