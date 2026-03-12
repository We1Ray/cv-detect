#!/usr/bin/env python3
"""
========================================================================
  cv-detect 全功能測試套件 — 按模組與方法分類
========================================================================

測試策略:
  - 每個模組的每個公開方法/函式都有至少一個測試
  - 使用合成影像 (256x256) 避免外部檔案依賴
  - 按照「分類 → 方法 → 執行順序」組織
  - 輸出帶有 PASS / FAIL 標記的詳細報告

分類順序 (由底層到高層):
  A. 共用工具 (shared)
  B. 設定模組 (config)
  C. 影像前處理 (preprocessor)
  D. 深度學習核心 (autoencoder, dataset)
  E. 異常評分 (anomaly_scorer)
  F. HALCON 影像處理運算子 (halcon_ops) — 16 個子分類
  G. 區域運算 (region, region_ops)
  H. 配方系統 (recipe)
  I. 訓練管線 (trainer)
  J. 推論管線 (inference)
  K. 視覺化 (heatmap, report, training_plots)
========================================================================
"""

import sys
import os
import time
import json
import math
import traceback
import tempfile
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Any, Optional

# 確保專案路徑
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "dl_anomaly"))

import numpy as np
import cv2
import torch

# =======================================================================
# 測試基礎設施
# =======================================================================

@dataclass
class TestResult:
    category: str
    subcategory: str
    method: str
    passed: bool
    elapsed_ms: float
    error: str = ""
    order: int = 0


ALL_RESULTS: List[TestResult] = []
_order_counter = 0


def run_test(category: str, subcategory: str, method: str, func, *args, **kwargs):
    """執行一個測試並記錄結果"""
    global _order_counter
    _order_counter += 1
    t0 = time.perf_counter()
    try:
        result = func(*args, **kwargs)
        elapsed = (time.perf_counter() - t0) * 1000
        ALL_RESULTS.append(TestResult(
            category=category, subcategory=subcategory, method=method,
            passed=True, elapsed_ms=elapsed, order=_order_counter
        ))
        return result
    except Exception as e:
        elapsed = (time.perf_counter() - t0) * 1000
        ALL_RESULTS.append(TestResult(
            category=category, subcategory=subcategory, method=method,
            passed=False, elapsed_ms=elapsed,
            error=f"{type(e).__name__}: {e}",
            order=_order_counter
        ))
        return None


# =======================================================================
# 合成測試影像
# =======================================================================

def make_gray_image(h=256, w=256) -> np.ndarray:
    """建立合成灰階測試影像 (含漸層、圓形、矩形)"""
    img = np.zeros((h, w), dtype=np.uint8)
    # 水平漸層背景
    for x in range(w):
        img[:, x] = int(x / w * 200) + 30
    # 白色圓形
    cv2.circle(img, (w // 2, h // 2), 50, 255, -1)
    # 灰色矩形
    cv2.rectangle(img, (30, 30), (100, 100), 180, -1)
    # 小噪點
    np.random.seed(42)
    noise = np.random.randint(0, 20, (h, w), dtype=np.uint8)
    img = cv2.add(img, noise)
    return img


def make_color_image(h=256, w=256) -> np.ndarray:
    """建立合成 BGR 彩色測試影像"""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = make_gray_image(h, w)  # Blue channel
    img[:, :, 1] = np.roll(make_gray_image(h, w), 50, axis=1)  # Green
    img[:, :, 2] = np.roll(make_gray_image(h, w), -50, axis=0)  # Red
    return img


def make_binary_mask(h=256, w=256) -> np.ndarray:
    """建立二值遮罩"""
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (128, 128), 40, 255, -1)
    cv2.circle(mask, (60, 60), 20, 255, -1)
    cv2.rectangle(mask, (180, 180), (230, 230), 255, -1)
    return mask


# 預建測試影像
GRAY_IMG = make_gray_image()
COLOR_IMG = make_color_image()
BINARY_MASK = make_binary_mask()
GRAY_IMG_2 = np.roll(GRAY_IMG, 10, axis=1)  # 偏移版本 (用於雙影像運算)
COLOR_IMG_2 = np.roll(COLOR_IMG, 10, axis=1)


# =======================================================================
# A. 共用工具 (shared)
# =======================================================================

def test_shared():
    cat = "A.共用工具"
    sub = "validation"

    from shared.validation import (
        validate_image, validate_kernel_size, validate_positive,
        validate_range, ImageValidationError
    )

    # validate_image — 正常
    run_test(cat, sub, "validate_image(正常灰階)", validate_image, GRAY_IMG)
    run_test(cat, sub, "validate_image(正常彩色)", validate_image, COLOR_IMG)

    # validate_image — 異常
    def test_validate_none():
        try:
            validate_image(None)
            raise AssertionError("Should raise")
        except ImageValidationError:
            pass
    run_test(cat, sub, "validate_image(None→error)", test_validate_none)

    def test_validate_empty():
        try:
            validate_image(np.array([]))
            raise AssertionError("Should raise")
        except ImageValidationError:
            pass
    run_test(cat, sub, "validate_image(empty→error)", test_validate_empty)

    # validate_kernel_size
    run_test(cat, sub, "validate_kernel_size(3)", validate_kernel_size, 3)
    def test_even_kernel():
        try:
            validate_kernel_size(4)
            raise AssertionError("Should raise")
        except ImageValidationError:
            pass
    run_test(cat, sub, "validate_kernel_size(偶數→error)", test_even_kernel)

    # validate_positive
    run_test(cat, sub, "validate_positive(1.0)", validate_positive, 1.0)
    def test_neg():
        try:
            validate_positive(-1)
            raise AssertionError("Should raise")
        except ImageValidationError:
            pass
    run_test(cat, sub, "validate_positive(負數→error)", test_neg)

    # validate_range
    run_test(cat, sub, "validate_range(0.5, 0, 1)", validate_range, 0.5, 0, 1)

    # op_logger
    sub = "op_logger"
    import logging
    from shared.op_logger import log_operation
    test_logger = logging.getLogger("test")

    @log_operation(test_logger)
    def dummy_op(x):
        return x * 2
    run_test(cat, sub, "log_operation 裝飾器", dummy_op, 5)


# =======================================================================
# B. 設定模組 (config)
# =======================================================================

def test_config():
    cat = "B.設定模組"
    sub = "Config"

    from dl_anomaly.config import Config, _resolve_path, _parse_bool, _select_device

    # 輔助函式
    run_test(cat, sub, "_resolve_path(相對路徑)", _resolve_path, "./test")
    run_test(cat, sub, "_parse_bool('true')", lambda: assert_eq(_parse_bool("true"), True))
    run_test(cat, sub, "_parse_bool('false')", lambda: assert_eq(_parse_bool("false"), False))
    run_test(cat, sub, "_select_device('cpu')", lambda: assert_eq(_select_device("cpu"), "cpu"))

    # Config 建構
    def test_config_init():
        cfg = Config()
        assert cfg.in_channels in (1, 3)
        assert cfg.image_size > 0
        return cfg
    cfg = run_test(cat, sub, "Config.__init__()", test_config_init)

    # to_dict / from_dict
    if cfg:
        def test_to_dict():
            d = cfg.to_dict()
            assert isinstance(d, dict)
            assert "image_size" in d
            return d
        d = run_test(cat, sub, "Config.to_dict()", test_to_dict)

        if d:
            def test_from_dict():
                cfg2 = Config.from_dict(d)
                assert cfg2.image_size == cfg.image_size
            run_test(cat, sub, "Config.from_dict()", test_from_dict)


# =======================================================================
# C. 影像前處理 (preprocessor)
# =======================================================================

def test_preprocessor():
    cat = "C.影像前處理"
    sub = "ImagePreprocessor"

    from dl_anomaly.core.preprocessor import ImagePreprocessor

    pp = run_test(cat, sub, "__init__(size=256, gray=False)", ImagePreprocessor, 256, False)
    pp_gray = run_test(cat, sub, "__init__(size=256, gray=True)", ImagePreprocessor, 256, True)

    if pp:
        # get_transforms
        def test_eval_transform():
            t = pp.get_transforms(augment=False)
            assert t is not None
            return t
        run_test(cat, sub, "get_transforms(augment=False)", test_eval_transform)

        def test_train_transform():
            t = pp.get_transforms(augment=True)
            assert t is not None
        run_test(cat, sub, "get_transforms(augment=True)", test_train_transform)

        # 使用臨時檔案測試 load_and_preprocess
        tmp_dir = tempfile.mkdtemp()
        tmp_img_path = os.path.join(tmp_dir, "test.png")
        cv2.imwrite(tmp_img_path, COLOR_IMG)

        def test_load():
            tensor = pp.load_and_preprocess(tmp_img_path)
            assert tensor.shape[0] == 3
            assert tensor.shape[1] == 256
            return tensor
        tensor = run_test(cat, sub, "load_and_preprocess()", test_load)

        if tensor is not None:
            def test_inverse():
                arr = pp.inverse_normalize(tensor)
                assert arr.dtype == np.uint8
                assert arr.ndim in (2, 3)
            run_test(cat, sub, "inverse_normalize()", test_inverse)

        shutil.rmtree(tmp_dir, ignore_errors=True)

    if pp_gray:
        def test_gray_transforms():
            t = pp_gray.get_transforms(augment=False)
            assert t is not None
        run_test(cat, sub, "get_transforms(灰階模式)", test_gray_transforms)


# =======================================================================
# D. 深度學習核心
# =======================================================================

def test_autoencoder():
    cat = "D.深度學習核心"
    sub = "ConvBlock"

    from dl_anomaly.core.autoencoder import ConvBlock, Encoder, Decoder, AnomalyAutoencoder

    # ConvBlock
    def test_convblock():
        block = ConvBlock(3, 32, residual=True)
        x = torch.randn(1, 3, 64, 64)
        out = block(x)
        assert out.shape == (1, 32, 64, 64)
    run_test(cat, sub, "ConvBlock(3→32, residual=True).forward()", test_convblock)

    def test_convblock_no_res():
        block = ConvBlock(32, 32, residual=False)
        x = torch.randn(1, 32, 32, 32)
        out = block(x)
        assert out.shape == (1, 32, 32, 32)
    run_test(cat, sub, "ConvBlock(residual=False).forward()", test_convblock_no_res)

    # Encoder
    sub = "Encoder"
    def test_encoder():
        enc = Encoder(3, 32, 4, 128)
        x = torch.randn(1, 3, 256, 256)
        z = enc(x)
        assert z.shape == (1, 128)
        return enc
    run_test(cat, sub, "Encoder(3,32,4,128).forward()", test_encoder)

    # Decoder
    sub = "Decoder"
    def test_decoder():
        dec = Decoder(128, 32, 4, 3, 256)
        z = torch.randn(1, 128)
        out = dec(z)
        assert out.shape == (1, 3, 256, 256)
    run_test(cat, sub, "Decoder(128,32,4,3,256).forward()", test_decoder)

    # AnomalyAutoencoder
    sub = "AnomalyAutoencoder"
    def test_ae_forward():
        model = AnomalyAutoencoder(3, 128, 32, 4, 256)
        x = torch.randn(1, 3, 256, 256)
        out = model(x)
        assert out.shape == x.shape
        return model
    model = run_test(cat, sub, "forward()", test_ae_forward)

    if model:
        def test_get_latent():
            x = torch.randn(1, 3, 256, 256)
            z = model.get_latent(x)
            assert z.shape == (1, 128)
        run_test(cat, sub, "get_latent()", test_get_latent)

    # _init_weights 已在建構時呼叫，驗證參數非零
    def test_init_weights():
        m = AnomalyAutoencoder(3, 64, 16, 2, 128)
        has_nonzero = any(p.abs().sum() > 0 for p in m.parameters())
        assert has_nonzero
    run_test(cat, sub, "_init_weights()", test_init_weights)


def test_dataset():
    cat = "D.深度學習核心"
    sub = "DefectFreeDataset"

    from dl_anomaly.core.dataset import DefectFreeDataset

    # 建立臨時目錄和影像
    tmp_dir = tempfile.mkdtemp()
    for i in range(3):
        cv2.imwrite(os.path.join(tmp_dir, f"img_{i}.png"), GRAY_IMG)

    def test_init():
        ds = DefectFreeDataset(tmp_dir)
        assert len(ds) == 3
        return ds
    ds = run_test(cat, sub, "__init__(含3張圖)", test_init)

    if ds:
        def test_len():
            assert len(ds) == 3
        run_test(cat, sub, "__len__()", test_len)

        def test_getitem():
            item, path = ds[0]
            assert isinstance(path, str)
        run_test(cat, sub, "__getitem__(0)", test_getitem)

    # 空目錄
    empty_dir = tempfile.mkdtemp()
    def test_empty():
        ds_empty = DefectFreeDataset(empty_dir)
        assert len(ds_empty) == 0
    run_test(cat, sub, "__init__(空目錄)", test_empty)

    shutil.rmtree(tmp_dir, ignore_errors=True)
    shutil.rmtree(empty_dir, ignore_errors=True)


# =======================================================================
# E. 異常評分
# =======================================================================

def test_anomaly_scorer():
    cat = "E.異常評分"
    sub = "AnomalyScorer"

    from dl_anomaly.core.anomaly_scorer import AnomalyScorer

    scorer = run_test(cat, sub, "__init__(cpu)", AnomalyScorer, "cpu")

    if scorer:
        # 準備測試資料
        orig = GRAY_IMG.copy()
        recon = cv2.GaussianBlur(orig, (5, 5), 2)  # 模擬重建（略有模糊）

        # compute_pixel_error
        def test_pixel_error():
            err = scorer.compute_pixel_error(orig, recon)
            assert err.shape == orig.shape[:2]
            assert err.dtype == np.float32
            assert 0 <= err.max() <= 1
            return err
        err = run_test(cat, sub, "compute_pixel_error(灰階)", test_pixel_error)

        # compute_pixel_error 彩色
        orig_c = COLOR_IMG.copy()
        recon_c = cv2.GaussianBlur(orig_c, (5, 5), 2)
        def test_pixel_error_color():
            err = scorer.compute_pixel_error(orig_c, recon_c)
            assert err.ndim == 2
        run_test(cat, sub, "compute_pixel_error(彩色)", test_pixel_error_color)

        # compute_ssim_map
        def test_ssim_gray():
            ssim = scorer.compute_ssim_map(orig, recon)
            assert ssim.shape == orig.shape[:2]
            assert ssim.dtype == np.float32
            return ssim
        run_test(cat, sub, "compute_ssim_map(灰階)", test_ssim_gray)

        def test_ssim_color():
            ssim = scorer.compute_ssim_map(orig_c, recon_c)
            assert ssim.ndim == 2
        run_test(cat, sub, "compute_ssim_map(彩色)", test_ssim_color)

        # compute_combined_error
        def test_combined():
            comb = scorer.compute_combined_error(orig, recon, ssim_weight=0.5)
            assert comb.shape == orig.shape[:2]
        run_test(cat, sub, "compute_combined_error(weight=0.5)", test_combined)

        def test_combined_no_ssim():
            comb = scorer.compute_combined_error(orig, recon, ssim_weight=0.0)
            assert comb.shape == orig.shape[:2]
        run_test(cat, sub, "compute_combined_error(weight=0.0)", test_combined_no_ssim)

        # compute_image_score
        if err is not None:
            def test_img_score():
                score = scorer.compute_image_score(err)
                assert isinstance(score, float)
                return score
            score = run_test(cat, sub, "compute_image_score()", test_img_score)

        # fit_threshold
        def test_fit():
            training_errors = [0.01, 0.02, 0.015, 0.03, 0.025, 0.018]
            th = scorer.fit_threshold(training_errors, 95.0)
            assert isinstance(th, float)
            assert th > 0
        run_test(cat, sub, "fit_threshold()", test_fit)

        # fit_threshold 空列表
        def test_fit_empty():
            try:
                scorer.fit_threshold([], 95.0)
                raise AssertionError("Should raise")
            except ValueError:
                pass
        run_test(cat, sub, "fit_threshold(空列表→error)", test_fit_empty)

        # classify
        def test_classify():
            result = scorer.classify(0.05)
            assert isinstance(result, bool)
        run_test(cat, sub, "classify()", test_classify)

        # classify 未 fit
        def test_classify_no_fit():
            s2 = AnomalyScorer("cpu")
            try:
                s2.classify(0.5)
                raise AssertionError("Should raise")
            except RuntimeError:
                pass
        run_test(cat, sub, "classify(未fit→error)", test_classify_no_fit)

        # create_anomaly_map
        if err is not None:
            def test_anomaly_map():
                amap = scorer.create_anomaly_map(err, gaussian_sigma=4.0)
                assert amap.shape == err.shape
                assert 0 <= amap.max() <= 1
            run_test(cat, sub, "create_anomaly_map()", test_anomaly_map)


# =======================================================================
# F. HALCON 影像處理運算子 — 16 子分類
# =======================================================================

def test_halcon_ops():
    cat = "F.HALCON運算子"

    from dl_anomaly.core import halcon_ops as hops

    gray = GRAY_IMG.copy()
    color = COLOR_IMG.copy()
    gray2 = GRAY_IMG_2.copy()
    color2 = COLOR_IMG_2.copy()

    # =============================================
    # F1. 影像算術
    # =============================================
    sub = "F1.影像算術"
    run_test(cat, sub, "add_image()", hops.add_image, gray, gray2, 0.5, 0)
    run_test(cat, sub, "sub_image()", hops.sub_image, gray, gray2, 1.0, 128)
    run_test(cat, sub, "mult_image()", hops.mult_image, gray, gray2, 1.0, 0)
    run_test(cat, sub, "abs_image()", hops.abs_image, gray)
    run_test(cat, sub, "invert_image()", hops.invert_image, gray)
    run_test(cat, sub, "scale_image(mult=1.2, add=10)", hops.scale_image, gray, 1.2, 10)
    run_test(cat, sub, "log_image(base=e)", hops.log_image, gray, "e")
    run_test(cat, sub, "log_image(base=2)", hops.log_image, gray, "2")
    run_test(cat, sub, "log_image(base=10)", hops.log_image, gray, "10")
    run_test(cat, sub, "exp_image(base=e)", hops.exp_image, gray, "e")
    run_test(cat, sub, "exp_image(base=2)", hops.exp_image, gray, "2")
    run_test(cat, sub, "gamma_image(γ=0.5)", hops.gamma_image, gray, 0.5)
    run_test(cat, sub, "gamma_image(γ=2.0)", hops.gamma_image, gray, 2.0)
    run_test(cat, sub, "min_image()", hops.min_image, gray, gray2)
    run_test(cat, sub, "max_image()", hops.max_image, gray, gray2)
    run_test(cat, sub, "crop_rectangle()", hops.crop_rectangle, gray, 10, 10, 100, 100)

    # =============================================
    # F1b. 域操作
    # =============================================
    sub = "F1b.域操作"
    region = hops.Region(mask=BINARY_MASK.copy())
    run_test(cat, sub, "reduce_domain()", hops.reduce_domain, gray, region, 0)
    run_test(cat, sub, "crop_domain()", hops.crop_domain, gray, region)

    # =============================================
    # F2. 濾波器
    # =============================================
    sub = "F2.濾波器"
    run_test(cat, sub, "mean_image(k=5)", hops.mean_image, gray, 5)
    run_test(cat, sub, "median_image(k=5)", hops.median_image, gray, 5)
    run_test(cat, sub, "gauss_filter(σ=1.5)", hops.gauss_filter, gray, 1.5)
    run_test(cat, sub, "gauss_blur(k=5)", hops.gauss_blur, gray, 5)
    run_test(cat, sub, "binomial_filter(k=5)", hops.binomial_filter, gray, 5)
    run_test(cat, sub, "bilateral_filter(d=9)", hops.bilateral_filter, gray, 9, 75, 75)
    run_test(cat, sub, "sharpen_image(0.5)", hops.sharpen_image, gray, 0.5)
    run_test(cat, sub, "emphasize(k=7,f=1.5)", hops.emphasize, gray, 7, 1.5)
    run_test(cat, sub, "laplace_filter()", hops.laplace_filter, gray)
    run_test(cat, sub, "sobel_filter(both)", hops.sobel_filter, gray, "both")
    run_test(cat, sub, "sobel_filter(x)", hops.sobel_filter, gray, "x")
    run_test(cat, sub, "sobel_filter(y)", hops.sobel_filter, gray, "y")
    run_test(cat, sub, "prewitt_filter()", hops.prewitt_filter, gray)
    run_test(cat, sub, "derivative_gauss(σ=1,x)", hops.derivative_gauss, gray, 1.0, "x")
    run_test(cat, sub, "derivative_gauss(σ=1,y)", hops.derivative_gauss, gray, 1.0, "y")

    # =============================================
    # F3. 邊緣偵測
    # =============================================
    sub = "F3.邊緣偵測"
    run_test(cat, sub, "edges_canny(50,150)", hops.edges_canny, gray, 50, 150, 1.0)
    run_test(cat, sub, "edges_sobel(both)", hops.edges_sobel, gray, "both")
    run_test(cat, sub, "zero_crossing()", hops.zero_crossing, gray)

    # =============================================
    # F4. 灰度形態學
    # =============================================
    sub = "F4.灰度形態學"
    run_test(cat, sub, "gray_erosion(k=5)", hops.gray_erosion, gray, 5)
    run_test(cat, sub, "gray_dilation(k=5)", hops.gray_dilation, gray, 5)
    run_test(cat, sub, "gray_opening(k=5)", hops.gray_opening, gray, 5)
    run_test(cat, sub, "gray_closing(k=5)", hops.gray_closing, gray, 5)
    run_test(cat, sub, "top_hat(k=9)", hops.top_hat, gray, 9)
    run_test(cat, sub, "bottom_hat(k=9)", hops.bottom_hat, gray, 9)
    run_test(cat, sub, "gray_opening_shape(rect,k=5)", hops.gray_opening_shape, gray, 5, 5, "rect")
    run_test(cat, sub, "gray_closing_shape(ellipse,k=5)", hops.gray_closing_shape, gray, 5, 5, "ellipse")
    gray_blurred = cv2.GaussianBlur(gray, (15, 15), 5)
    run_test(cat, sub, "dyn_threshold(offset=10)", hops.dyn_threshold, gray, gray_blurred, 10)
    run_test(cat, sub, "var_threshold(k=15,factor=0.2)", hops.var_threshold, gray, 15, 0.2)
    run_test(cat, sub, "local_threshold(k=15,bias=10)", hops.local_threshold, gray, 15, 10)

    # =============================================
    # F5. FFT / 頻率域
    # =============================================
    sub = "F5.FFT/頻率域"
    run_test(cat, sub, "fft_image()", hops.fft_image, gray)
    run_test(cat, sub, "gen_gauss_filter((256,256),σ=30)", hops.gen_gauss_filter, (256, 256), 30)
    run_test(cat, sub, "freq_filter(lowpass)", hops.freq_filter, gray, "lowpass", 30)
    run_test(cat, sub, "freq_filter(highpass)", hops.freq_filter, gray, "highpass", 30)

    # =============================================
    # F6. 幾何變換
    # =============================================
    sub = "F6.幾何變換"
    run_test(cat, sub, "rotate_image(90°)", hops.rotate_image, gray, 90)
    run_test(cat, sub, "rotate_image(180°)", hops.rotate_image, gray, 180)
    run_test(cat, sub, "rotate_image(45°)", hops.rotate_image, gray, 45)
    run_test(cat, sub, "mirror_image(horizontal)", hops.mirror_image, gray, "horizontal")
    run_test(cat, sub, "mirror_image(vertical)", hops.mirror_image, gray, "vertical")
    run_test(cat, sub, "zoom_image(0.5,0.5)", hops.zoom_image, gray, 0.5, 0.5)
    run_test(cat, sub, "zoom_image(2.0,2.0)", hops.zoom_image, gray, 2.0, 2.0)

    # affine_trans_image
    M_affine = np.float32([[1, 0, 10], [0, 1, 20]])
    run_test(cat, sub, "affine_trans_image()", hops.affine_trans_image, gray, M_affine)

    # projective_trans_image
    pts1 = np.float32([[0, 0], [255, 0], [0, 255], [255, 255]])
    pts2 = np.float32([[10, 10], [245, 5], [5, 250], [250, 250]])
    M_proj = cv2.getPerspectiveTransform(pts1, pts2)
    run_test(cat, sub, "projective_trans_image()", hops.projective_trans_image, gray, M_proj)

    # polar_trans_image
    run_test(cat, sub, "polar_trans_image()", hops.polar_trans_image, gray, 128, 128)

    # =============================================
    # F7. 色彩空間
    # =============================================
    sub = "F7.色彩空間"
    run_test(cat, sub, "rgb_to_gray()", hops.rgb_to_gray, color)
    run_test(cat, sub, "rgb_to_hsv()", hops.rgb_to_hsv, color)
    run_test(cat, sub, "rgb_to_hls()", hops.rgb_to_hls, color)

    def test_decompose3():
        r, g, b = hops.decompose3(color)
        assert r.ndim == 2
        return r, g, b
    channels = run_test(cat, sub, "decompose3()", test_decompose3)

    if channels:
        run_test(cat, sub, "compose3()", hops.compose3, channels[0], channels[1], channels[2])

    run_test(cat, sub, "histogram_eq(灰階)", hops.histogram_eq, gray)
    run_test(cat, sub, "histogram_eq(彩色)", hops.histogram_eq, color)
    run_test(cat, sub, "illuminate(k=41,gain=1.0)", hops.illuminate, gray, 41, 1.0)

    # =============================================
    # F8. 紋理 / 特徵影像
    # =============================================
    sub = "F8.紋理/特徵"
    run_test(cat, sub, "texture_laws(L5E5)", hops.texture_laws, gray, "L5E5")
    run_test(cat, sub, "texture_laws(E5S5)", hops.texture_laws, gray, "E5S5")
    run_test(cat, sub, "entropy_image(k=5)", hops.entropy_image, gray, 5)
    run_test(cat, sub, "deviation_image(k=5)", hops.deviation_image, gray, 5)
    run_test(cat, sub, "local_min(k=3)", hops.local_min, gray, 3)
    run_test(cat, sub, "local_max(k=3)", hops.local_max, gray, 3)
    run_test(cat, sub, "mean_curvature()", hops.mean_curvature, gray)

    # =============================================
    # F9. 輪廓 (XLD)
    # =============================================
    sub = "F9.輪廓(XLD)"
    def test_find_contours():
        contours = hops.find_contours(BINARY_MASK)
        assert isinstance(contours, list)
        return contours
    contours = run_test(cat, sub, "find_contours()", test_find_contours)

    if contours and len(contours) > 0:
        cnt = contours[0]
        if len(cnt) >= 5:
            run_test(cat, sub, "fit_line()", hops.fit_line, cnt)
            run_test(cat, sub, "fit_circle()", hops.fit_circle, cnt)
            run_test(cat, sub, "fit_ellipse()", hops.fit_ellipse, cnt)
        run_test(cat, sub, "contour_length()", hops.contour_length, cnt)
        run_test(cat, sub, "convex_hull()", hops.convex_hull, cnt)
        def test_select():
            result = hops.select_contours(contours, "length", 10, 100000)
            assert isinstance(result, list)
        run_test(cat, sub, "select_contours(length,10,100000)", test_select)

    # =============================================
    # F10. 匹配
    # =============================================
    sub = "F10.匹配"
    template = gray[80:180, 80:180].copy()
    run_test(cat, sub, "template_match_ncc()", hops.template_match_ncc, gray, template, 0.6)

    # match_shape — img and template
    run_test(cat, sub, "match_shape()", hops.match_shape, gray, template, 3)

    run_test(cat, sub, "match_template(ccoeff_normed)", hops.match_template, gray, template, "ccoeff_normed")

    # =============================================
    # F11. 量測
    # =============================================
    sub = "F11.量測"
    def test_line_profile():
        profile = hops.measure_line_profile(gray, 10, 10, 200, 200)
        assert isinstance(profile, dict)
    run_test(cat, sub, "measure_line_profile()", test_line_profile)

    def test_distance_pp():
        d = hops.distance_pp(0, 0, 3, 4)
        assert abs(d - 5.0) < 0.01
    run_test(cat, sub, "distance_pp()", test_distance_pp)

    def test_angle_ll():
        angle = hops.angle_ll((1, 0, 0, 0), (0, 1, 0, 0))
        assert isinstance(angle, float)
    run_test(cat, sub, "angle_ll()", test_angle_ll)

    def test_area_center():
        result = hops.area_center(BINARY_MASK)
        assert isinstance(result, dict) or isinstance(result, tuple)
    run_test(cat, sub, "area_center()", test_area_center)

    # =============================================
    # F12. 繪圖
    # =============================================
    sub = "F12.繪圖"
    canvas = np.zeros((256, 256), dtype=np.uint8)
    img_shape = (256, 256)
    run_test(cat, sub, "gen_rectangle()", hops.gen_rectangle, 50, 50, 100, 100, img_shape)
    run_test(cat, sub, "gen_circle()", hops.gen_circle, 128, 128, 40, img_shape)
    run_test(cat, sub, "gen_ellipse()", hops.gen_ellipse, 128, 128, 60, 40, 0, img_shape)
    run_test(cat, sub, "gen_region_polygon()", hops.gen_region_polygon,
             [(50, 50), (200, 50), (200, 200), (50, 200)], img_shape)
    run_test(cat, sub, "paint_region()", hops.paint_region,
             hops.Region(mask=BINARY_MASK.copy()), canvas.copy(), (200, 200, 200))

    canvas_color = np.zeros((256, 256, 3), dtype=np.uint8)
    run_test(cat, sub, "draw_text()", hops.draw_text, canvas_color.copy(), "Test", 50, 50)
    run_test(cat, sub, "draw_line()", hops.draw_line, canvas_color.copy(), 10, 10, 200, 200)
    run_test(cat, sub, "draw_rectangle()", hops.draw_rectangle, canvas_color.copy(), 30, 30, 100, 100)
    run_test(cat, sub, "draw_circle()", hops.draw_circle, canvas_color.copy(), 128, 128, 50)
    run_test(cat, sub, "draw_arrow()", hops.draw_arrow, canvas_color.copy(), 50, 50, 200, 200)
    run_test(cat, sub, "draw_cross()", hops.draw_cross, canvas_color.copy(), 128, 128)

    # =============================================
    # F13. 條碼 / QR Code
    # =============================================
    sub = "F13.條碼/QR"
    # 用空白影像測試 (不會找到條碼但不應崩潰)
    run_test(cat, sub, "find_barcode(無條碼)", hops.find_barcode, gray)
    run_test(cat, sub, "find_qrcode(無QR)", hops.find_qrcode, gray)
    run_test(cat, sub, "find_datamatrix(無DM)", hops.find_datamatrix, gray)

    # =============================================
    # F14. 分割
    # =============================================
    sub = "F14.分割"
    run_test(cat, sub, "watersheds()", hops.watersheds, gray)
    run_test(cat, sub, "distance_transform(L2)", hops.distance_transform, BINARY_MASK)
    run_test(cat, sub, "skeleton()", hops.skeleton, BINARY_MASK)

    # =============================================
    # F15. 特徵點
    # =============================================
    sub = "F15.特徵點"
    def test_harris():
        pts = hops.points_harris(gray)
        assert isinstance(pts, (list, np.ndarray))
    run_test(cat, sub, "points_harris()", test_harris)

    def test_shi_tomasi():
        pts = hops.points_shi_tomasi(gray)
        assert isinstance(pts, (list, np.ndarray))
    run_test(cat, sub, "points_shi_tomasi()", test_shi_tomasi)

    # =============================================
    # F16. Hough 變換
    # =============================================
    sub = "F16.Hough變換"
    edge_img = cv2.Canny(gray, 50, 150)
    run_test(cat, sub, "hough_lines()", hops.hough_lines, edge_img)
    run_test(cat, sub, "hough_circles()", hops.hough_circles, gray)

    # =============================================
    # F17. 雜項
    # =============================================
    sub = "F17.雜項"
    run_test(cat, sub, "optical_flow()", hops.optical_flow, gray, gray2)
    run_test(cat, sub, "gen_gauss_pyramid(levels=3)", hops.gen_gauss_pyramid, gray, 3)
    run_test(cat, sub, "estimate_noise()", hops.estimate_noise, gray)
    run_test(cat, sub, "abs_diff_image()", hops.abs_diff_image, gray, gray2)
    run_test(cat, sub, "clahe(clip=2.0)", hops.clahe, gray)

    # halcon_ops.Region __post_init__
    sub = "F0.Region類別"
    def test_halcon_region():
        r = hops.Region(mask=BINARY_MASK.copy())
        assert r.area > 0
        assert r.cx > 0
    run_test(cat, sub, "Region.__post_init__()", test_halcon_region)


# =======================================================================
# G. 區域運算 (region, region_ops)
# =======================================================================

def test_region():
    cat = "G.區域運算"

    from dl_anomaly.core.region import Region, RegionProperties
    from dl_anomaly.core.region_ops import (
        threshold, binary_threshold, connection,
        select_shape, compute_region_properties,
        region_to_display_image, _ensure_gray_uint8, _mask_to_region
    )

    gray = GRAY_IMG.copy()

    # threshold
    sub = "閾值操作"
    def test_manual_threshold():
        r = threshold(gray, 100, 200)
        assert isinstance(r, Region)
        assert r.num_regions >= 0
        return r
    region = run_test(cat, sub, "threshold(100,200)", test_manual_threshold)

    def test_otsu():
        r = binary_threshold(gray, method="otsu")
        assert isinstance(r, Region)
        return r
    run_test(cat, sub, "binary_threshold(otsu)", test_otsu)

    def test_adaptive():
        r = binary_threshold(gray, method="adaptive", block_size=11, c_value=2)
        assert isinstance(r, Region)
    run_test(cat, sub, "binary_threshold(adaptive)", test_adaptive)

    # 彩色影像
    def test_threshold_color():
        r = threshold(COLOR_IMG.copy(), 50, 200)
        assert isinstance(r, Region)
    run_test(cat, sub, "threshold(彩色輸入)", test_threshold_color)

    # connection
    sub = "連通分析"
    if region:
        def test_connection():
            r = connection(region)
            assert r.num_regions >= 0
            assert len(r.properties) >= 0
            return r
        conn_region = run_test(cat, sub, "connection()", test_connection)
    else:
        conn_region = None

    # select_shape
    sub = "形狀篩選"
    if conn_region and conn_region.num_regions > 0:
        def test_select_shape():
            r = select_shape(conn_region, "area", 100, 100000)
            assert isinstance(r, Region)
        run_test(cat, sub, "select_shape(area)", test_select_shape)

        def test_select_circ():
            r = select_shape(conn_region, "circularity", 0.3, 1.0)
            assert isinstance(r, Region)
        run_test(cat, sub, "select_shape(circularity)", test_select_circ)

    # compute_region_properties
    sub = "屬性計算"
    def test_compute_props():
        # 建立簡單 label 影像
        labels = np.zeros((256, 256), dtype=np.int32)
        cv2.circle(labels, (128, 128), 40, 1, -1)
        cv2.circle(labels, (60, 60), 20, 2, -1)
        props = compute_region_properties(labels, gray)
        assert len(props) >= 2
        for p in props:
            assert p.area > 0
            assert p.perimeter >= 0
        return props
    props = run_test(cat, sub, "compute_region_properties()", test_compute_props)

    # RegionProperties 欄位
    if props:
        def test_props_fields():
            p = props[0]
            _ = p.index, p.area, p.centroid, p.bbox
            _ = p.width, p.height, p.circularity
            _ = p.rectangularity, p.aspect_ratio, p.compactness
            _ = p.convexity, p.perimeter, p.orientation
            _ = p.mean_value, p.min_value, p.max_value
        run_test(cat, sub, "RegionProperties 欄位存取", test_props_fields)

    # Region 方法
    sub = "Region方法"
    if region:
        def test_binary_mask():
            m = region.to_binary_mask()
            assert m.dtype == np.uint8
            assert set(np.unique(m)).issubset({0, 255})
        run_test(cat, sub, "to_binary_mask()", test_binary_mask)

        def test_color_mask():
            m = region.to_color_mask()
            assert m.ndim == 3
        run_test(cat, sub, "to_color_mask()", test_color_mask)

    if conn_region and conn_region.num_regions > 0:
        def test_single():
            r = conn_region.get_single_region(1)
            assert isinstance(r, Region)
        run_test(cat, sub, "get_single_region(1)", test_single)

        def test_filter_by():
            r = conn_region.filter_by("area", 10, 100000)
            assert isinstance(r, Region)
        run_test(cat, sub, "filter_by(area)", test_filter_by)

    # 集合運算
    sub = "集合運算"
    if region:
        region2 = threshold(gray, 50, 150)
        if region2:
            def test_union():
                r = region.union(region2)
                assert isinstance(r, Region)
            run_test(cat, sub, "union()", test_union)

            def test_intersection():
                r = region.intersection(region2)
                assert isinstance(r, Region)
            run_test(cat, sub, "intersection()", test_intersection)

            def test_difference():
                r = region.difference(region2)
                assert isinstance(r, Region)
            run_test(cat, sub, "difference()", test_difference)

            def test_complement():
                r = region.complement()
                assert isinstance(r, Region)
            run_test(cat, sub, "complement()", test_complement)

    # region_to_display_image
    sub = "視覺化"
    if conn_region:
        def test_display():
            img = region_to_display_image(conn_region, gray)
            assert img.ndim == 3
            assert img.dtype == np.uint8
        run_test(cat, sub, "region_to_display_image()", test_display)

        def test_display_no_annot():
            img = region_to_display_image(
                conn_region, gray,
                show_labels=False, show_bbox=False, show_cross=False
            )
            assert img.ndim == 3
        run_test(cat, sub, "region_to_display_image(無標註)", test_display_no_annot)

    # 內部輔助
    sub = "內部輔助"
    def test_ensure_gray():
        g = _ensure_gray_uint8(COLOR_IMG)
        assert g.ndim == 2
    run_test(cat, sub, "_ensure_gray_uint8(彩色)", test_ensure_gray)

    def test_ensure_gray_float():
        f = np.random.rand(100, 100).astype(np.float32)
        g = _ensure_gray_uint8(f)
        assert g.dtype == np.uint8
    run_test(cat, sub, "_ensure_gray_uint8(float)", test_ensure_gray_float)

    def test_mask_to_region():
        r = _mask_to_region(BINARY_MASK.copy(), gray)
        assert isinstance(r, Region)
    run_test(cat, sub, "_mask_to_region()", test_mask_to_region)


# =======================================================================
# H. 配方系統 (recipe)
# =======================================================================

def test_recipe():
    cat = "H.配方系統"
    sub = "Recipe"

    from dl_anomaly.core.recipe import Recipe, replay_recipe

    # 建構
    def test_init():
        r = Recipe()
        assert r.steps == []
        assert r.version == 1
        return r
    recipe = run_test(cat, sub, "Recipe.__init__()", test_init)

    # save / load
    tmp_dir = tempfile.mkdtemp()
    recipe_path = os.path.join(tmp_dir, "test_recipe.json")

    steps = [
        {"name": "灰階", "category": "quick_op", "op": "grayscale", "params": {}},
        {"name": "高斯模糊", "category": "halcon", "op": "gauss_blur", "params": {"ksize": 5}},
        {"name": "邊緣", "category": "quick_op", "op": "edge", "params": {}},
    ]

    def test_save():
        r = Recipe(steps=steps)
        r.save(recipe_path)
        assert os.path.exists(recipe_path)
    run_test(cat, sub, "Recipe.save()", test_save)

    def test_load():
        r = Recipe.load(recipe_path)
        assert len(r.steps) == 3
        assert r.steps[0]["op"] == "grayscale"
        return r
    loaded = run_test(cat, sub, "Recipe.load()", test_load)

    # replay_recipe
    sub = "replay"
    if loaded:
        def test_replay():
            results = replay_recipe(loaded, COLOR_IMG.copy())
            assert len(results) == 3
            for name, img, _ in results:
                assert isinstance(img, np.ndarray)
        run_test(cat, sub, "replay_recipe(3步)", test_replay)

    # _replay_halcon 各操作
    sub = "replay_halcon"
    halcon_ops = [
        ("rgb_to_gray", {}),
        ("invert_image", {}),
        ("mean_image", {"ksize": 5}),
        ("gauss_blur", {"ksize": 5}),
        ("sobel_filter", {}),
    ]
    for op, params in halcon_ops:
        test_recipe_obj = Recipe(steps=[
            {"name": op, "category": "halcon", "op": op, "params": params}
        ])
        run_test(cat, sub, f"replay({op})", replay_recipe, test_recipe_obj, COLOR_IMG.copy())

    # _replay_quick_op
    sub = "replay_quick_op"
    for op in ["grayscale", "blur", "edge", "histeq", "invert"]:
        test_recipe_obj = Recipe(steps=[
            {"name": op, "category": "quick_op", "op": op, "params": {}}
        ])
        run_test(cat, sub, f"replay({op})", replay_recipe, test_recipe_obj, COLOR_IMG.copy())

    shutil.rmtree(tmp_dir, ignore_errors=True)


# =======================================================================
# I. 訓練管線 (僅測試非訓練方法)
# =======================================================================

def test_trainer():
    cat = "I.訓練管線"
    sub = "TrainingPipeline"

    from dl_anomaly.config import Config
    from dl_anomaly.pipeline.trainer import (
        TrainingPipeline, ssim_loss,
        _gaussian_kernel_1d, _gaussian_kernel_2d
    )

    # 輔助函式
    def test_gauss_1d():
        k = _gaussian_kernel_1d(11, 1.5)
        assert k.shape == (11,)
        assert abs(k.sum().item() - 1.0) < 0.01
    run_test(cat, sub, "_gaussian_kernel_1d(11,1.5)", test_gauss_1d)

    def test_gauss_2d():
        k = _gaussian_kernel_2d(11, 1.5, 3)
        assert k.shape == (3, 1, 11, 11)
    run_test(cat, sub, "_gaussian_kernel_2d(11,1.5,3)", test_gauss_2d)

    # ssim_loss
    def test_ssim_loss():
        x = torch.randn(2, 3, 64, 64)
        y = torch.randn(2, 3, 64, 64)
        loss = ssim_loss(x, y, window_size=7)
        assert loss.ndim == 0  # scalar
        assert loss.item() >= 0
    run_test(cat, sub, "ssim_loss()", test_ssim_loss)

    # TrainingPipeline 建構
    def test_tp_init():
        cfg = Config()
        tp = TrainingPipeline(cfg)
        assert tp.model is None
        return tp
    tp = run_test(cat, sub, "TrainingPipeline.__init__()", test_tp_init)

    if tp:
        # build_model
        def test_build():
            m = tp.build_model()
            assert isinstance(m, torch.nn.Module)
            return m
        model = run_test(cat, sub, "build_model()", test_build)

        # compute_loss
        if model:
            def test_loss():
                x = torch.randn(1, 3, 256, 256)
                y = torch.randn(1, 3, 256, 256)
                loss = tp.compute_loss(x, y)
                assert loss.ndim == 0
            run_test(cat, sub, "compute_loss()", test_loss)

        # request_stop
        def test_stop():
            tp.request_stop()
            assert tp._stop_requested is True
        run_test(cat, sub, "request_stop()", test_stop)

    # save_checkpoint / load_checkpoint
    sub = "Checkpoint"
    if tp:
        tp.model = tp.build_model()
        tp.optimizer = torch.optim.AdamW(tp.model.parameters(), lr=0.001)
        tmp_dir = tempfile.mkdtemp()
        ckpt_path = Path(tmp_dir) / "test_ckpt.pt"

        def test_save():
            tp.save_checkpoint(ckpt_path, epoch=1, loss=0.5)
            assert ckpt_path.exists()
        run_test(cat, sub, "save_checkpoint()", test_save)

        def test_load():
            model, cfg, state = TrainingPipeline.load_checkpoint(ckpt_path, "cpu")
            assert isinstance(model, torch.nn.Module)
            assert state["epoch"] == 1
        run_test(cat, sub, "load_checkpoint()", test_load)

        shutil.rmtree(tmp_dir, ignore_errors=True)


# =======================================================================
# J. 推論管線
# =======================================================================

def test_inference():
    cat = "J.推論管線"
    sub = "InferencePipeline"

    from dl_anomaly.config import Config
    from dl_anomaly.pipeline.inference import InferencePipeline, InspectionResult
    from dl_anomaly.pipeline.trainer import TrainingPipeline

    # 建立臨時 checkpoint
    cfg = Config()
    tp = TrainingPipeline(cfg)
    tp.model = tp.build_model()
    tp.optimizer = torch.optim.AdamW(tp.model.parameters(), lr=0.001)

    tmp_dir = tempfile.mkdtemp()
    ckpt_path = Path(tmp_dir) / "test_model.pt"
    tp.save_checkpoint(ckpt_path, epoch=1, loss=0.5)
    # 加入 threshold
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state["threshold"] = 0.05
    torch.save(state, ckpt_path)

    # InferencePipeline 建構
    def test_init():
        pipe = InferencePipeline(ckpt_path, device="cpu")
        assert pipe.scorer.threshold == 0.05
        return pipe
    pipe = run_test(cat, sub, "InferencePipeline.__init__()", test_init)

    # 建立測試圖片
    test_img_path = Path(tmp_dir) / "test_input.png"
    cv2.imwrite(str(test_img_path), COLOR_IMG)

    if pipe:
        # inspect_single
        def test_inspect():
            result = pipe.inspect_single(test_img_path)
            assert isinstance(result, InspectionResult)
            assert result.original.dtype == np.uint8
            assert result.error_map.dtype == np.float32
            assert isinstance(result.anomaly_score, float)
            assert isinstance(result.is_defective, bool)
            return result
        result = run_test(cat, sub, "inspect_single()", test_inspect)

        # inspect_batch
        def test_batch():
            results = pipe.inspect_batch(tmp_dir)
            assert isinstance(results, list)
            assert len(results) >= 1
        run_test(cat, sub, "inspect_batch()", test_batch)

        # _create_defect_mask
        if result:
            def test_defect_mask():
                mask = InferencePipeline._create_defect_mask(result.error_map)
                assert mask.dtype == np.uint8
            run_test(cat, sub, "_create_defect_mask()", test_defect_mask)

            def test_extract_regions():
                regions = InferencePipeline._extract_defect_regions(result.defect_mask)
                assert isinstance(regions, list)
            run_test(cat, sub, "_extract_defect_regions()", test_extract_regions)

    # InspectionResult dataclass
    sub = "InspectionResult"
    def test_dataclass():
        r = InspectionResult(
            original=GRAY_IMG, reconstruction=GRAY_IMG,
            error_map=np.zeros((256, 256), dtype=np.float32),
            defect_mask=np.zeros((256, 256), dtype=np.uint8),
            anomaly_score=0.01, is_defective=False,
        )
        assert r.defect_regions == []
    run_test(cat, sub, "InspectionResult建構", test_dataclass)

    shutil.rmtree(tmp_dir, ignore_errors=True)


# =======================================================================
# K. 視覺化
# =======================================================================

def test_visualization():
    cat = "K.視覺化"

    # K1. heatmap
    sub = "K1.heatmap"
    from dl_anomaly.visualization.heatmap import (
        create_error_heatmap, create_defect_overlay,
        create_reconstruction_comparison, create_composite_result
    )

    error_map = np.random.rand(256, 256).astype(np.float32)

    def test_heatmap():
        h = create_error_heatmap(error_map)
        assert h.shape == (256, 256, 3)
        assert h.dtype == np.uint8
    run_test(cat, sub, "create_error_heatmap()", test_heatmap)

    def test_heatmap_viridis():
        h = create_error_heatmap(error_map, colormap=cv2.COLORMAP_VIRIDIS)
        assert h.shape == (256, 256, 3)
    run_test(cat, sub, "create_error_heatmap(VIRIDIS)", test_heatmap_viridis)

    rgb_img = cv2.cvtColor(COLOR_IMG, cv2.COLOR_BGR2RGB)
    def test_overlay():
        o = create_defect_overlay(rgb_img, error_map, threshold=0.5, alpha=0.5)
        assert o.shape == rgb_img.shape
    run_test(cat, sub, "create_defect_overlay()", test_overlay)

    def test_overlay_gray():
        o = create_defect_overlay(GRAY_IMG, error_map, threshold=0.3)
        assert o.ndim == 3
    run_test(cat, sub, "create_defect_overlay(灰階)", test_overlay_gray)

    def test_comparison():
        c = create_reconstruction_comparison(rgb_img, rgb_img)
        assert c.shape[1] > rgb_img.shape[1]
    run_test(cat, sub, "create_reconstruction_comparison()", test_comparison)

    def test_composite():
        mask = (error_map > 0.5).astype(np.uint8) * 255
        heatmap = create_error_heatmap(error_map)
        c = create_composite_result(rgb_img, rgb_img, heatmap, mask)
        assert c.ndim == 3
    run_test(cat, sub, "create_composite_result()", test_composite)

    # K2. report
    sub = "K2.report"
    from dl_anomaly.visualization.report import generate_summary_stats, save_result_image
    from dl_anomaly.pipeline.inference import InspectionResult

    def test_summary_empty():
        stats = generate_summary_stats([])
        assert stats["total"] == 0
    run_test(cat, sub, "generate_summary_stats(空列表)", test_summary_empty)

    results_list = [
        InspectionResult(
            original=rgb_img, reconstruction=rgb_img,
            error_map=error_map, defect_mask=BINARY_MASK,
            anomaly_score=0.05, is_defective=False
        ),
        InspectionResult(
            original=rgb_img, reconstruction=rgb_img,
            error_map=error_map, defect_mask=BINARY_MASK,
            anomaly_score=0.15, is_defective=True,
            defect_regions=[{"id": 1, "area": 500}]
        ),
    ]

    def test_summary():
        stats = generate_summary_stats(results_list)
        assert stats["total"] == 2
        assert stats["defective"] == 1
        assert stats["pass"] == 1
        assert 0 < stats["mean_score"] < 1
    run_test(cat, sub, "generate_summary_stats(2筆)", test_summary)

    # save_result_image
    tmp_dir = tempfile.mkdtemp()
    def test_save_result():
        p = save_result_image(results_list[0], "test_img.png", tmp_dir)
        assert p.exists()
    run_test(cat, sub, "save_result_image()", test_save_result)
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # K3. training_plots
    sub = "K3.training_plots"
    from dl_anomaly.visualization.training_plots import (
        plot_loss_curve, plot_reconstruction_samples, plot_error_distribution
    )

    def test_loss_curve():
        fig = plot_loss_curve([0.5, 0.3, 0.2], [0.6, 0.4, 0.3])
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)
    run_test(cat, sub, "plot_loss_curve()", test_loss_curve)

    def test_recon_samples():
        imgs = [GRAY_IMG, GRAY_IMG, GRAY_IMG]
        fig = plot_reconstruction_samples(imgs, imgs, n=3)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)
    run_test(cat, sub, "plot_reconstruction_samples()", test_recon_samples)

    def test_error_dist():
        fig = plot_error_distribution([0.01, 0.02, 0.05, 0.1, 0.15], threshold=0.08)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)
    run_test(cat, sub, "plot_error_distribution()", test_error_dist)

    def test_error_dist_no_thresh():
        fig = plot_error_distribution([0.01, 0.02, 0.05])
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)
    run_test(cat, sub, "plot_error_distribution(無閾值)", test_error_dist_no_thresh)


# =======================================================================
# 輔助
# =======================================================================

def assert_eq(a, b):
    assert a == b, f"Expected {b}, got {a}"


# =======================================================================
# 執行全部測試 & 輸出報告
# =======================================================================

def print_report():
    """Output categorized test report"""
    import io
    import sys as _sys
    # Force UTF-8 output
    _sys.stdout = io.TextIOWrapper(_sys.stdout.buffer, encoding='utf-8', errors='replace')

    print("\n" + "=" * 90)
    print("  cv-detect Full Function Test Report")
    print("=" * 90)

    # Group by category
    categories = {}
    for r in ALL_RESULTS:
        key = (r.category, r.subcategory)
        categories.setdefault(key, []).append(r)

    total_pass = sum(1 for r in ALL_RESULTS if r.passed)
    total_fail = sum(1 for r in ALL_RESULTS if not r.passed)
    total = len(ALL_RESULTS)

    current_cat = ""
    for (cat, sub), tests in sorted(categories.items()):
        if cat != current_cat:
            current_cat = cat
            print(f"\n{'-' * 90}")
            print(f"  [{cat}]")
            print(f"{'-' * 90}")

        cat_pass = sum(1 for t in tests if t.passed)
        cat_total = len(tests)
        print(f"\n  >> {sub}  ({cat_pass}/{cat_total})")

        for t in tests:
            status = "[PASS]" if t.passed else "[FAIL]"
            time_str = f"{t.elapsed_ms:7.1f}ms"
            print(f"    {status}  {time_str}  #{t.order:03d}  {t.method}")
            if not t.passed:
                print(f"            -> {t.error}")

    # Summary
    print(f"\n{'=' * 90}")
    print(f"  Summary: {total} tests | {total_pass} passed | {total_fail} failed")
    if total_fail == 0:
        print("  Result: ALL PASSED!")
    else:
        print(f"  Result: {total_fail} FAILED")
    print(f"{'=' * 90}")

    # Category stats
    print(f"\n  Category Statistics:")
    print(f"  {'Category':<25s} {'Pass':>6s} {'Fail':>6s} {'Total':>6s} {'Rate':>8s}")
    print(f"  {'-' * 55}")
    cat_stats = {}
    for r in ALL_RESULTS:
        cat_stats.setdefault(r.category, {"pass": 0, "fail": 0})
        if r.passed:
            cat_stats[r.category]["pass"] += 1
        else:
            cat_stats[r.category]["fail"] += 1

    for cat_name in sorted(cat_stats):
        s = cat_stats[cat_name]
        t = s["pass"] + s["fail"]
        rate = s["pass"] / t * 100 if t > 0 else 0
        print(f"  {cat_name:<25s} {s['pass']:>6d} {s['fail']:>6d} {t:>6d} {rate:>7.1f}%")

    # Execution order
    print(f"\n{'=' * 90}")
    print("  Method Execution Order")
    print(f"{'=' * 90}")
    print(f"  {'#':>4s}  {'Status':<8s} {'Subcategory':<20s} {'Method'}")
    print(f"  {'-' * 80}")
    for r in sorted(ALL_RESULTS, key=lambda x: x.order):
        status = "PASS" if r.passed else "FAIL"
        print(f"  {r.order:>4d}  {status:<8s} {r.subcategory:<20s} {r.method}")

    return total_fail


def main():
    print("開始執行 cv-detect 全功能測試...")
    print(f"Python: {sys.version}")
    print(f"NumPy: {np.__version__}")
    print(f"OpenCV: {cv2.__version__}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {'可用' if torch.cuda.is_available() else '不可用'}")

    t0 = time.time()

    # 按順序執行各分類測試
    test_shared()
    test_config()
    test_preprocessor()
    test_autoencoder()
    test_dataset()
    test_anomaly_scorer()
    test_halcon_ops()
    test_region()
    test_recipe()
    test_trainer()
    test_inference()
    test_visualization()

    elapsed = time.time() - t0
    print(f"\n測試耗時: {elapsed:.1f} 秒")

    fail_count = print_report()
    return fail_count


if __name__ == "__main__":
    fail_count = main()
    sys.exit(1 if fail_count > 0 else 0)
