"""
gui/app.py - 向後相容包裝器

將 MainApp 對應到新的 InspectorApp，維持 main.py 的匯入路徑不變。
"""

from gui.inspector_app import InspectorApp as MainApp

__all__ = ["MainApp"]
