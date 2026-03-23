"""
config.py - 專案組態管理模組

從 .env 檔案載入所有參數，並以 dataclass 提供型別安全的存取介面。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import dotenv_values


def _parse_bool(value: str) -> bool:
    """將字串轉換為布林值。"""
    return value.strip().lower() in ("true", "1", "yes", "on")


@dataclass
class Config:
    """統一管理 Variation Model 所有參數的組態類別。"""

    # ── 影像路徑 ──
    train_image_dir: Path = field(default_factory=lambda: Path("data") / "train")
    test_image_dir: Path = field(default_factory=lambda: Path("data") / "test")
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

    def __post_init__(self):
        if self.target_width <= 0:
            raise ValueError(f"target_width must be positive, got {self.target_width}")
        if self.target_height <= 0:
            raise ValueError(f"target_height must be positive, got {self.target_height}")
        if self.abs_threshold < 0:
            raise ValueError(f"abs_threshold must be non-negative, got {self.abs_threshold}")
        if self.var_threshold < 0:
            raise ValueError(f"var_threshold must be non-negative, got {self.var_threshold}")
        if self.morph_kernel_size < 0:
            raise ValueError(f"morph_kernel_size must be non-negative, got {self.morph_kernel_size}")

    @classmethod
    def from_env(cls, env_path: Optional[str] = None) -> "Config":
        """從 .env 檔案載入組態，未設定的參數使用預設值。

        Args:
            env_path: .env 檔案路徑；若為 None 則自動偵測專案根目錄。

        Returns:
            解析完成的 Config 實例。
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

        # ── 前處理 ──
        if raw.get("TARGET_WIDTH"):
            kwargs["target_width"] = int(raw["TARGET_WIDTH"])
        if raw.get("TARGET_HEIGHT"):
            kwargs["target_height"] = int(raw["TARGET_HEIGHT"])
        if raw.get("GRAYSCALE"):
            kwargs["grayscale"] = _parse_bool(raw["GRAYSCALE"])
        if raw.get("GAUSSIAN_BLUR_KERNEL"):
            kwargs["gaussian_blur_kernel"] = int(raw["GAUSSIAN_BLUR_KERNEL"])
        if raw.get("ENABLE_ALIGNMENT"):
            kwargs["enable_alignment"] = _parse_bool(raw["ENABLE_ALIGNMENT"])
        if raw.get("ALIGNMENT_METHOD"):
            kwargs["alignment_method"] = raw["ALIGNMENT_METHOD"].strip().lower()

        # ── Variation Model 參數 ──
        if raw.get("ABS_THRESHOLD"):
            kwargs["abs_threshold"] = int(raw["ABS_THRESHOLD"])
        if raw.get("VAR_THRESHOLD"):
            kwargs["var_threshold"] = float(raw["VAR_THRESHOLD"])

        # ── 形態學清理 ──
        if raw.get("MORPH_KERNEL_SIZE"):
            kwargs["morph_kernel_size"] = int(raw["MORPH_KERNEL_SIZE"])
        if raw.get("MIN_DEFECT_AREA"):
            kwargs["min_defect_area"] = int(raw["MIN_DEFECT_AREA"])

        # ── 多尺度 ──
        if raw.get("ENABLE_MULTISCALE"):
            kwargs["enable_multiscale"] = _parse_bool(raw["ENABLE_MULTISCALE"])
        if raw.get("SCALE_LEVELS"):
            kwargs["scale_levels"] = int(raw["SCALE_LEVELS"])

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

    def update(self, **kwargs) -> "Config":
        """回傳一份套用更新後的新 Config（不可變更新）。"""
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
        return Config(**current)
