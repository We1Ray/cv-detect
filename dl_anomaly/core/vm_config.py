"""
core/vm_config.py - Variation Model 組態管理模組

從 .env 檔案載入所有參數，並以 dataclass 提供型別安全的存取介面。
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import dotenv_values

logger = logging.getLogger(__name__)


def _parse_bool(value: str) -> bool:
    """將字串轉換為布林值。"""
    return value.strip().lower() in ("true", "1", "yes", "on")


@dataclass
class VMConfig:
    """統一管理 Variation Model 所有參數的組態類別。"""

    # ── 影像路徑 ──
    train_image_dir: Path = Path(".")
    test_image_dir: Path = Path(".")
    reference_image: Optional[Path] = None

    # ── 模型持久化 ──
    model_save_dir: Path = Path("./models")
    results_dir: Path = Path("./results")

    # ── 前處理 ──
    target_width: int = 640
    target_height: int = 480
    grayscale: bool = True
    gaussian_blur_kernel: int = 3
    enable_alignment: bool = True
    alignment_method: str = "ecc"

    # ── Variation Model 參數 ──
    abs_threshold: int = 10
    var_threshold: float = 3.0

    # ── 形態學清理 ──
    morph_kernel_size: int = 3
    min_defect_area: int = 50

    # ── 多尺度 ──
    enable_multiscale: bool = True
    scale_levels: int = 3

    @classmethod
    def from_env(cls, env_path: Optional[str] = None) -> "VMConfig":
        """從 .env 檔案載入組態，未設定的參數使用預設值。

        Args:
            env_path: .env 檔案路徑；若為 None 則自動偵測專案根目錄。

        Returns:
            解析完成的 VMConfig 實例。
        """
        if env_path is None:
            env_path = str(Path(__file__).parent / ".env")

        raw = dotenv_values(env_path)

        kwargs: dict = {}

        # ── 影像路徑 ──
        if raw.get("TRAIN_IMAGE_DIR"):
            kwargs["train_image_dir"] = Path(raw["TRAIN_IMAGE_DIR"])
        if raw.get("TEST_IMAGE_DIR"):
            kwargs["test_image_dir"] = Path(raw["TEST_IMAGE_DIR"])
        if raw.get("REFERENCE_IMAGE"):
            kwargs["reference_image"] = Path(raw["REFERENCE_IMAGE"])

        # ── 模型持久化 ──
        if raw.get("MODEL_SAVE_DIR"):
            kwargs["model_save_dir"] = Path(raw["MODEL_SAVE_DIR"])
        if raw.get("RESULTS_DIR"):
            kwargs["results_dir"] = Path(raw["RESULTS_DIR"])

        def _safe_int(key: str) -> int | None:
            val = raw.get(key)
            if not val:
                return None
            try:
                return int(val)
            except (ValueError, TypeError):
                logger.warning("無法將 %s='%s' 解析為整數，使用預設值", key, val)
                return None

        def _safe_float(key: str) -> float | None:
            val = raw.get(key)
            if not val:
                return None
            try:
                return float(val)
            except (ValueError, TypeError):
                logger.warning("無法將 %s='%s' 解析為浮點數，使用預設值", key, val)
                return None

        # ── 前處理 ──
        v = _safe_int("TARGET_WIDTH")
        if v is not None:
            kwargs["target_width"] = v
        v = _safe_int("TARGET_HEIGHT")
        if v is not None:
            kwargs["target_height"] = v
        if raw.get("GRAYSCALE"):
            kwargs["grayscale"] = _parse_bool(raw["GRAYSCALE"])
        v = _safe_int("GAUSSIAN_BLUR_KERNEL")
        if v is not None:
            kwargs["gaussian_blur_kernel"] = v
        if raw.get("ENABLE_ALIGNMENT"):
            kwargs["enable_alignment"] = _parse_bool(raw["ENABLE_ALIGNMENT"])
        if raw.get("ALIGNMENT_METHOD"):
            kwargs["alignment_method"] = raw["ALIGNMENT_METHOD"].strip().lower()

        # ── Variation Model 參數 ──
        v = _safe_int("ABS_THRESHOLD")
        if v is not None:
            kwargs["abs_threshold"] = v
        vf = _safe_float("VAR_THRESHOLD")
        if vf is not None:
            kwargs["var_threshold"] = vf

        # ── 形態學清理 ──
        v = _safe_int("MORPH_KERNEL_SIZE")
        if v is not None:
            kwargs["morph_kernel_size"] = v
        v = _safe_int("MIN_DEFECT_AREA")
        if v is not None:
            kwargs["min_defect_area"] = v

        # ── 多尺度 ──
        if raw.get("ENABLE_MULTISCALE"):
            kwargs["enable_multiscale"] = _parse_bool(raw["ENABLE_MULTISCALE"])
        v = _safe_int("SCALE_LEVELS")
        if v is not None:
            kwargs["scale_levels"] = v

        return cls(**kwargs)

    def to_dict(self) -> dict:
        """將組態轉換為字典，便於序列化或顯示。"""
        return {
            "train_image_dir": str(self.train_image_dir),
            "test_image_dir": str(self.test_image_dir),
            "reference_image": str(self.reference_image) if self.reference_image else "",
            "model_save_dir": str(self.model_save_dir),
            "results_dir": str(self.results_dir),
            "target_width": self.target_width,
            "target_height": self.target_height,
            "grayscale": self.grayscale,
            "gaussian_blur_kernel": self.gaussian_blur_kernel,
            "enable_alignment": self.enable_alignment,
            "alignment_method": self.alignment_method,
            "abs_threshold": self.abs_threshold,
            "var_threshold": self.var_threshold,
            "morph_kernel_size": self.morph_kernel_size,
            "min_defect_area": self.min_defect_area,
            "enable_multiscale": self.enable_multiscale,
            "scale_levels": self.scale_levels,
        }

    def update(self, **kwargs) -> "VMConfig":
        """回傳一份套用更新後的新 VMConfig（不可變更新）。"""
        current = self.to_dict()
        current.update(kwargs)
        # 需要將字串路徑轉回 Path
        for key in ("train_image_dir", "test_image_dir", "reference_image",
                     "model_save_dir", "results_dir"):
            val = current.get(key)
            if val and isinstance(val, str) and val.strip():
                current[key] = Path(val)
            elif key == "reference_image" and (not val or not str(val).strip()):
                current[key] = None
        return VMConfig(**current)
