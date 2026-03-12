"""
gui/app.py - 向後相容包裝器

將 MainApp 對應到新的 HalconApp，維持 main.py 的匯入路徑不變。
"""

from gui.halcon_app import HalconApp as MainApp

__all__ = ["MainApp"]
