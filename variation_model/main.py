"""
main.py - Variation Model Inspector 應用程式進入點

負責：
1. 設定 sys.path 以支援模組匯入
2. 設定日誌系統
3. 載入組態
4. 啟動 HALCON HDevelop 風格 GUI 主視窗
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_sys_path() -> None:
    """將專案根目錄加入 sys.path，確保所有模組可正確匯入。"""
    project_root = str(Path(__file__).resolve().parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def setup_logging() -> None:
    """設定全域日誌格式與等級。"""
    log_format = "[%(asctime)s] %(levelname)-8s %(name)s: %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


def main() -> None:
    """應用程式主進入點。"""
    # 1. 設定 sys.path
    setup_sys_path()

    # 2. 設定日誌
    setup_logging()

    logger = logging.getLogger(__name__)
    logger.info("Variation Model Inspector (HALCON Style) starting...")

    # 3. 載入組態
    from config import Config

    try:
        config = Config.from_env()
        logger.info("Configuration loaded successfully")
    except Exception as exc:
        logger.warning("Failed to load .env, using defaults: %s", exc)
        config = Config()

    # 4. 確保輸出目錄存在
    Path(config.model_save_dir).mkdir(parents=True, exist_ok=True)
    Path(config.results_dir).mkdir(parents=True, exist_ok=True)

    # 5. 啟動 HALCON 風格 GUI
    from gui.halcon_app import HalconApp

    app = HalconApp(config)
    logger.info("HALCON-style GUI initialized, entering main loop")
    app.mainloop()
    logger.info("Application closed")


if __name__ == "__main__":
    main()
