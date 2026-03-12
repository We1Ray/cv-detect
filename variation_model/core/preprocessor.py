"""
core/preprocessor.py - 影像前處理模組

負責影像載入、縮放、灰階轉換、平滑、對齊等前處理流程。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

from config import Config

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """影像前處理器：提供載入、縮放、灰階、平滑、對齊等功能。"""

    def __init__(self, config: Config) -> None:
        self.config = config

    # ------------------------------------------------------------------ #
    #  載入影像                                                           #
    # ------------------------------------------------------------------ #
    def load_image(self, path: Union[str, Path]) -> np.ndarray:
        """從磁碟載入影像，回傳 BGR 或灰階 NumPy 陣列。

        Args:
            path: 影像檔案路徑。

        Returns:
            載入的影像陣列。

        Raises:
            FileNotFoundError: 檔案不存在。
            IOError: OpenCV 無法讀取檔案。
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"影像檔案不存在: {path}")

        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise IOError(f"OpenCV 無法讀取影像: {path}")

        logger.debug("載入影像: %s  shape=%s", path.name, image.shape)
        return image

    # ------------------------------------------------------------------ #
    #  縮放                                                               #
    # ------------------------------------------------------------------ #
    def resize(self, image: np.ndarray) -> np.ndarray:
        """將影像縮放至組態中指定的目標尺寸。

        Args:
            image: 輸入影像。

        Returns:
            縮放後的影像。
        """
        target = (self.config.target_width, self.config.target_height)
        h, w = image.shape[:2]
        if (w, h) == target:
            return image
        resized = cv2.resize(image, target, interpolation=cv2.INTER_AREA)
        logger.debug("縮放: (%d, %d) -> (%d, %d)", w, h, target[0], target[1])
        return resized

    # ------------------------------------------------------------------ #
    #  灰階轉換                                                           #
    # ------------------------------------------------------------------ #
    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """若組態啟用灰階，則將 BGR 影像轉為灰階。

        Args:
            image: BGR 或已為灰階的影像。

        Returns:
            灰階影像（若組態未啟用則原樣返回）。
        """
        if not self.config.grayscale:
            return image
        if image.ndim == 2:
            return image
        if image.ndim == 3 and image.shape[2] == 1:
            return image[:, :, 0]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        logger.debug("灰階轉換完成")
        return gray

    # ------------------------------------------------------------------ #
    #  高斯平滑                                                           #
    # ------------------------------------------------------------------ #
    def smooth(self, image: np.ndarray) -> np.ndarray:
        """對影像施加高斯模糊以降低雜訊。

        Args:
            image: 輸入影像。

        Returns:
            平滑後的影像。
        """
        k = self.config.gaussian_blur_kernel
        if k <= 0:
            return image
        # 確保核心尺寸為奇數
        if k % 2 == 0:
            k += 1
        smoothed = cv2.GaussianBlur(image, (k, k), 0)
        logger.debug("高斯平滑: kernel=%d", k)
        return smoothed

    # ------------------------------------------------------------------ #
    #  影像對齊                                                           #
    # ------------------------------------------------------------------ #
    def align_to_reference(
        self, image: np.ndarray, reference: np.ndarray
    ) -> np.ndarray:
        """將影像對齊至參考影像。

        支援兩種方法：
        - ``ecc``: 基於 ECC 的多層金字塔對齊（cv2.findTransformECC，MOTION_EUCLIDEAN）
        - ``feature``: 基於 ORB 特徵點 + RANSAC 單應性矩陣的對齊

        Args:
            image: 待對齊影像。
            reference: 參考影像。

        Returns:
            對齊後的影像。
        """
        if not self.config.enable_alignment:
            return image

        method = self.config.alignment_method.lower()
        if method == "ecc":
            return self._align_ecc(image, reference)
        elif method == "feature":
            return self._align_feature(image, reference)
        else:
            logger.warning("未知的對齊方法 '%s'，跳過對齊", method)
            return image

    def _ensure_gray(self, image: np.ndarray) -> np.ndarray:
        """確保影像為灰階（用於對齊內部運算）。"""
        if image.ndim == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def _align_ecc(self, image: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """使用多層 ECC 金字塔進行亞像素級對齊。

        從最粗的金字塔層開始，逐層精煉仿射矩陣。
        """
        gray_img = self._ensure_gray(image)
        gray_ref = self._ensure_gray(reference)

        h, w = gray_ref.shape[:2]

        # 建立高斯金字塔（由細到粗）
        num_levels = 3
        pyr_img = [gray_img]
        pyr_ref = [gray_ref]
        for _ in range(num_levels - 1):
            pyr_img.append(cv2.pyrDown(pyr_img[-1]))
            pyr_ref.append(cv2.pyrDown(pyr_ref[-1]))

        # 初始化 2x3 歐幾里德仿射矩陣
        warp_matrix = np.eye(2, 3, dtype=np.float32)

        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            200,
            1e-6,
        )

        # 從最粗層向最細層迭代
        for level in range(num_levels - 1, -1, -1):
            try:
                _, warp_matrix = cv2.findTransformECC(
                    pyr_ref[level],
                    pyr_img[level],
                    warp_matrix,
                    cv2.MOTION_EUCLIDEAN,
                    criteria,
                )
            except cv2.error as exc:
                logger.warning("ECC 對齊在金字塔層 %d 失敗: %s", level, exc)
                return image

            # 放大仿射矩陣以匹配下一層解析度
            if level > 0:
                warp_matrix[0, 2] *= 2.0
                warp_matrix[1, 2] *= 2.0

        aligned = cv2.warpAffine(
            image,
            warp_matrix,
            (w, h),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE,
        )
        logger.debug("ECC 對齊完成")
        return aligned

    def _align_feature(
        self, image: np.ndarray, reference: np.ndarray
    ) -> np.ndarray:
        """使用 ORB 特徵點 + RANSAC 單應性矩陣進行對齊。"""
        gray_img = self._ensure_gray(image)
        gray_ref = self._ensure_gray(reference)

        h, w = reference.shape[:2] if reference.ndim == 2 else reference.shape[:2]

        orb = cv2.ORB_create(nfeatures=5000)
        kp1, des1 = orb.detectAndCompute(gray_img, None)
        kp2, des2 = orb.detectAndCompute(gray_ref, None)

        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            logger.warning("特徵點不足，無法進行特徵對齊")
            return image

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)

        # Lowe's ratio test
        good_matches = []
        for pair in matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        if len(good_matches) < 4:
            logger.warning("有效匹配點不足 (%d)，無法計算單應性矩陣", len(good_matches))
            return image

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if homography is None:
            logger.warning("無法計算單應性矩陣")
            return image

        aligned = cv2.warpPerspective(
            image,
            homography,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        logger.debug("特徵對齊完成 (匹配點: %d)", len(good_matches))
        return aligned

    # ------------------------------------------------------------------ #
    #  完整前處理流程                                                      #
    # ------------------------------------------------------------------ #
    def preprocess(
        self,
        path: Union[str, Path],
        reference: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """執行完整前處理流程：載入 -> 縮放 -> 灰階 -> 平滑 -> 對齊。

        Args:
            path: 影像檔案路徑。
            reference: 參考影像（用於對齊），為 None 則跳過對齊。

        Returns:
            前處理完畢的影像。
        """
        image = self.load_image(path)
        image = self.resize(image)
        image = self.to_grayscale(image)
        image = self.smooth(image)

        if reference is not None and self.config.enable_alignment:
            image = self.align_to_reference(image, reference)

        return image
