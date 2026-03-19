"""Backward-compatibility shim.

The original ``MainApp`` has been replaced by :class:`InspectorApp` in
``gui.inspector_app``.  This module re-exports it under the old name so that
any code that does ``from dl_anomaly.gui.app import MainApp`` continues
to work.
"""

from dl_anomaly.gui.inspector_app import InspectorApp as MainApp

__all__ = ["MainApp"]
