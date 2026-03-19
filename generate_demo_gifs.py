#!/usr/bin/env python3
"""Generate animated GIF demos for all inspection algorithms.

Produces GIFs demonstrating each detection pipeline on PCB defect images,
then updates README.md to embed them.

Usage:
    python generate_demo_gifs.py
"""

from __future__ import annotations

import sys
import os
import logging
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── Project path setup ────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("demo_gif")

# ── Paths ─────────────────────────────────────────────────────────────
PCB_DATASET = Path("/Users/WeiRay/Downloads/pcb defect.v3i.yolov7pytorch")
TRAIN_DIR = PCB_DATASET / "train" / "images"
TEST_DIR = PCB_DATASET / "test" / "images"
GIF_DIR = PROJECT_ROOT / "assets" / "demo"
GIF_DIR.mkdir(parents=True, exist_ok=True)

# ── Common constants ──────────────────────────────────────────────────
FRAME_SIZE = (480, 360)  # (width, height)
GIF_FPS = 2  # frames per second
GIF_DURATION = int(1000 / GIF_FPS)  # ms per frame
TITLE_HEIGHT = 36
FONT_SIZE = 20
N_DEMO_IMAGES = 6


# ======================================================================
#  Helper utilities
# ======================================================================

def _get_font(size: int = FONT_SIZE):
    """Try to load a CJK font, fall back to default."""
    cjk_paths = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ]
    for fp in cjk_paths:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                continue
    return ImageFont.load_default()


FONT = _get_font()
FONT_SMALL = _get_font(14)


def _load_images(directory: Path, n: int = N_DEMO_IMAGES) -> List[np.ndarray]:
    """Load n images from directory, resized to FRAME_SIZE."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    paths = sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    )[:n]
    images = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is not None:
            img = cv2.resize(img, FRAME_SIZE)
            images.append(img)
    return images


def _add_title_bar(frame_rgb: np.ndarray, title: str, bg_color=(30, 30, 30)) -> np.ndarray:
    """Add a title bar on top of the frame (RGB input/output)."""
    h, w = frame_rgb.shape[:2]
    canvas = np.full((h + TITLE_HEIGHT, w, 3), bg_color, dtype=np.uint8)
    canvas[TITLE_HEIGHT:, :] = frame_rgb

    pil_img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil_img)
    bbox = draw.textbbox((0, 0), title, font=FONT)
    tw = bbox[2] - bbox[0]
    tx = (w - tw) // 2
    draw.text((tx, 6), title, fill=(255, 255, 255), font=FONT)
    return np.array(pil_img)


def _add_label(frame_rgb: np.ndarray, label: str, position="bottom") -> np.ndarray:
    """Add a small label to the frame."""
    pil_img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(pil_img)
    h, w = frame_rgb.shape[:2]

    bbox = draw.textbbox((0, 0), label, font=FONT_SMALL)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    if position == "bottom":
        y = h - th - 8
    else:
        y = 8
    x = w - tw - 10

    # Semi-transparent background
    draw.rectangle([x - 4, y - 2, x + tw + 4, y + th + 2], fill=(0, 0, 0))
    draw.text((x, y), label, fill=(0, 255, 100), font=FONT_SMALL)
    return np.array(pil_img)


def _to_rgb(img: np.ndarray) -> np.ndarray:
    """Ensure image is RGB uint8."""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _resize_to_frame(img: np.ndarray) -> np.ndarray:
    """Resize to standard frame size."""
    return cv2.resize(img, FRAME_SIZE)


def _make_side_by_side(left: np.ndarray, right: np.ndarray, gap: int = 4) -> np.ndarray:
    """Create a side-by-side image resized to FRAME_SIZE."""
    h = min(left.shape[0], right.shape[0])
    w = (FRAME_SIZE[0] - gap) // 2

    l = cv2.resize(left, (w, h))
    r = cv2.resize(right, (w, h))
    sep = np.full((h, gap, 3), 60, dtype=np.uint8)
    combined = np.concatenate([l, sep, r], axis=1)
    return cv2.resize(combined, FRAME_SIZE)


def _save_gif(frames: List[np.ndarray], path: Path, duration: int = GIF_DURATION):
    """Save list of RGB numpy arrays as animated GIF."""
    pil_frames = [Image.fromarray(f) for f in frames]
    # Quantize to reduce file size
    pil_frames = [f.quantize(colors=128, method=Image.Quantize.MEDIANCUT).convert("RGB") for f in pil_frames]
    pil_frames_p = [f.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for f in pil_frames]
    pil_frames_p[0].save(
        path,
        save_all=True,
        append_images=pil_frames_p[1:],
        duration=duration,
        loop=0,
        optimize=True,
    )
    size_kb = path.stat().st_size / 1024
    log.info(f"  Saved: {path.name} ({size_kb:.0f} KB, {len(frames)} frames)")


# ======================================================================
#  1. Edge Detection Demo
# ======================================================================

def generate_edge_detection_gif(images: List[np.ndarray]) -> Path:
    """Canny / Sobel edge detection on PCB images."""
    log.info("Generating: Edge Detection GIF")
    from dl_anomaly.core.halcon_ops import edges_canny, sobel_filter

    frames = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rgb_orig = _to_rgb(img)

        # Canny edges
        canny = edges_canny(img, low=50, high=150, sigma=1.0)
        canny_rgb = cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB)
        # Colorize edges: green on dark
        canny_color = rgb_orig.copy()
        canny_color[canny > 0] = [0, 255, 100]

        # Frame 1: Original
        f1 = _add_title_bar(rgb_orig, "Edge Detection - PCB 邊緣檢測")
        f1 = _add_label(f1, "Original", "bottom")
        frames.append(f1)

        # Frame 2: Canny overlay
        f2 = _add_title_bar(canny_color, "Edge Detection - PCB 邊緣檢測")
        f2 = _add_label(f2, "Canny Edges", "bottom")
        frames.append(f2)

        # Sobel
        sobel = sobel_filter(img, direction="both")
        sobel_norm = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        sobel_color = cv2.applyColorMap(sobel_norm, cv2.COLORMAP_INFERNO)
        sobel_rgb = cv2.cvtColor(sobel_color, cv2.COLOR_BGR2RGB)

        f3 = _add_title_bar(sobel_rgb, "Edge Detection - PCB 邊緣檢測")
        f3 = _add_label(f3, "Sobel Magnitude", "bottom")
        frames.append(f3)

    path = GIF_DIR / "01_edge_detection.gif"
    _save_gif(frames, path, duration=600)
    return path


# ======================================================================
#  2. Threshold + Blob Analysis Demo
# ======================================================================

def generate_blob_analysis_gif(images: List[np.ndarray]) -> Path:
    """Threshold segmentation + connected component analysis."""
    log.info("Generating: Blob Analysis GIF")
    from dl_anomaly.core.region_ops import binary_threshold, connection, region_to_display_image

    frames = []
    for img in images:
        rgb_orig = _to_rgb(img)

        # Binary threshold (Otsu)
        region = binary_threshold(img, method="otsu")
        mask_rgb = cv2.cvtColor(region.to_binary_mask(), cv2.COLOR_GRAY2RGB)

        # Connected components
        conn = connection(region)

        # Display with labels and bbox
        display = region_to_display_image(
            conn, source_image=img,
            show_labels=True, show_bbox=True, show_cross=True, alpha=0.45,
        )
        display_rgb = _to_rgb(display)

        # Frame 1: Original
        f1 = _add_title_bar(_resize_to_frame(rgb_orig), "Blob Analysis - 連通區域分析")
        f1 = _add_label(f1, "Original")
        frames.append(f1)

        # Frame 2: Binary mask
        f2 = _add_title_bar(_resize_to_frame(mask_rgb), "Blob Analysis - 連通區域分析")
        f2 = _add_label(f2, "Otsu Threshold")
        frames.append(f2)

        # Frame 3: Labeled regions
        f3 = _add_title_bar(_resize_to_frame(display_rgb), "Blob Analysis - 連通區域分析")
        f3 = _add_label(f3, f"Regions: {conn.num_regions}")
        frames.append(f3)

    path = GIF_DIR / "02_blob_analysis.gif"
    _save_gif(frames, path, duration=700)
    return path


# ======================================================================
#  3. Morphological Operations Demo
# ======================================================================

def generate_morphology_gif(images: List[np.ndarray]) -> Path:
    """Morphological open/close/dilate/erode demo."""
    log.info("Generating: Morphology GIF")
    from dl_anomaly.core.halcon_ops import (
        edges_canny,
    )

    frames = []
    for img in images[:4]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        ops = [
            ("Original Binary", binary),
            ("Erode (侵蝕)", cv2.erode(binary, kernel, iterations=2)),
            ("Dilate (膨脹)", cv2.dilate(binary, kernel, iterations=2)),
            ("Open (開運算)", cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)),
            ("Close (閉運算)", cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)),
        ]

        for label, result in ops:
            result_rgb = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
            f = _add_title_bar(_resize_to_frame(result_rgb), "Morphology - 形態學運算")
            f = _add_label(f, label)
            frames.append(f)

    path = GIF_DIR / "03_morphology.gif"
    _save_gif(frames, path, duration=600)
    return path


# ======================================================================
#  4. FFT Frequency Domain Demo
# ======================================================================

def generate_fft_gif(images: List[np.ndarray]) -> Path:
    """FFT spectrum + frequency filtering demo."""
    log.info("Generating: FFT GIF")
    from shared.core.frequency import (
        compute_fft, inverse_fft,
        create_gaussian_filter, create_butterworth_filter,
    )

    frames = []
    for img in images[:4]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rgb_orig = _to_rgb(img)

        # Forward FFT
        fft_result = compute_fft(gray)

        # Magnitude spectrum (already uint8)
        mag_color = cv2.applyColorMap(fft_result.magnitude, cv2.COLORMAP_MAGMA)
        mag_rgb = cv2.cvtColor(mag_color, cv2.COLOR_BGR2RGB)

        # Gaussian lowpass filter
        h, w = fft_result.shape
        gauss_lp = create_gaussian_filter((h, w), sigma=30.0, filter_type="lowpass")
        filtered_spectrum = fft_result.complex_spectrum * gauss_lp
        filtered_fft = type(fft_result)(
            magnitude=fft_result.magnitude,
            phase=fft_result.phase,
            complex_spectrum=filtered_spectrum,
            shape=fft_result.shape,
        )
        reconstructed = inverse_fft(filtered_fft)
        recon_rgb = cv2.cvtColor(reconstructed, cv2.COLOR_GRAY2RGB)

        # Highpass = 1 - lowpass
        gauss_hp = create_gaussian_filter((h, w), sigma=30.0, filter_type="highpass")
        hp_spectrum = fft_result.complex_spectrum * gauss_hp
        hp_fft = type(fft_result)(
            magnitude=fft_result.magnitude,
            phase=fft_result.phase,
            complex_spectrum=hp_spectrum,
            shape=fft_result.shape,
        )
        hp_img = inverse_fft(hp_fft)
        hp_rgb = cv2.cvtColor(hp_img, cv2.COLOR_GRAY2RGB)

        # Frames
        f1 = _add_title_bar(_resize_to_frame(rgb_orig), "FFT - 頻域分析")
        f1 = _add_label(f1, "Original")
        frames.append(f1)

        f2 = _add_title_bar(_resize_to_frame(mag_rgb), "FFT - 頻域分析")
        f2 = _add_label(f2, "FFT Magnitude")
        frames.append(f2)

        f3 = _add_title_bar(_resize_to_frame(recon_rgb), "FFT - 頻域分析")
        f3 = _add_label(f3, "Lowpass Filtered")
        frames.append(f3)

        f4 = _add_title_bar(_resize_to_frame(hp_rgb), "FFT - 頻域分析")
        f4 = _add_label(f4, "Highpass (Defects)")
        frames.append(f4)

    path = GIF_DIR / "04_fft_frequency.gif"
    _save_gif(frames, path, duration=700)
    return path


# ======================================================================
#  5. Color Inspection Demo
# ======================================================================

def generate_color_inspect_gif(images: List[np.ndarray]) -> Path:
    """Color delta-E map + palette extraction demo."""
    log.info("Generating: Color Inspection GIF")
    from shared.core.color_inspect import (
        sample_color, compute_delta_e_map, build_color_palette,
    )

    frames = []
    for img in images[:4]:
        rgb_orig = _to_rgb(img)

        # Sample reference color from center region
        h, w = img.shape[:2]
        roi = (w // 4, h // 4, w // 2, h // 2)
        ref_sample = sample_color(img, roi=roi)

        # Delta-E map
        de_map = compute_delta_e_map(img, ref_sample.lab, method="CIE76")
        de_norm = np.clip(de_map / max(de_map.max(), 1e-6), 0, 1)
        de_u8 = (de_norm * 255).astype(np.uint8)
        de_color = cv2.applyColorMap(de_u8, cv2.COLORMAP_JET)
        de_rgb = cv2.cvtColor(de_color, cv2.COLOR_BGR2RGB)

        # Color palette
        palette = build_color_palette(img, n_colors=6)
        palette_bar = np.zeros((60, FRAME_SIZE[0], 3), dtype=np.uint8)
        pw = FRAME_SIZE[0] // len(palette)
        for idx, cs in enumerate(palette):
            r, g, b = cs.rgb
            palette_bar[:, idx * pw:(idx + 1) * pw] = [r, g, b]

        # Compose palette frame
        palette_frame = np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8)
        small_orig = cv2.resize(rgb_orig, (FRAME_SIZE[0], FRAME_SIZE[1] - 60))
        palette_frame[:FRAME_SIZE[1] - 60] = small_orig
        palette_frame[FRAME_SIZE[1] - 60:] = palette_bar

        # Frames
        f1 = _add_title_bar(_resize_to_frame(rgb_orig), "Color Inspection - 色彩檢測")
        f1 = _add_label(f1, "Original")
        frames.append(f1)

        f2 = _add_title_bar(_resize_to_frame(de_rgb), "Color Inspection - 色彩檢測")
        f2 = _add_label(f2, f"Delta-E Map (ref L*={ref_sample.lab[0]:.0f})")
        frames.append(f2)

        f3 = _add_title_bar(palette_frame, "Color Inspection - 色彩檢測")
        f3 = _add_label(f3, f"Palette ({len(palette)} colors)", "top")
        frames.append(f3)

    path = GIF_DIR / "05_color_inspection.gif"
    _save_gif(frames, path, duration=800)
    return path


# ======================================================================
#  6. Shape Matching Demo
# ======================================================================

def generate_shape_matching_gif(images: List[np.ndarray]) -> Path:
    """Template matching with gradient-based shape model."""
    log.info("Generating: Shape Matching GIF")
    from shared.core.shape_matching import create_shape_model, find_shape_model, draw_shape_matches

    # Use a small region from the first image as template
    template_src = images[0]
    h, w = template_src.shape[:2]
    # Crop a distinctive region (e.g., a component area)
    tx, ty, tw, th = w // 4, h // 4, w // 3, h // 3
    template = template_src[ty:ty + th, tx:tx + tw]

    model = create_shape_model(
        template,
        num_levels=3,
        angle_start=-0.1,
        angle_extent=0.2,
        min_contrast=20,
    )
    log.info(f"  Shape model created: {len(model.contour_points)} contour points")

    frames = []

    # Show template
    template_rgb = _to_rgb(template)
    template_display = np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8)
    t_resized = cv2.resize(template_rgb, (FRAME_SIZE[0] // 2, FRAME_SIZE[1] // 2))
    y_off = (FRAME_SIZE[1] - t_resized.shape[0]) // 2
    x_off = (FRAME_SIZE[0] - t_resized.shape[1]) // 2
    template_display[y_off:y_off + t_resized.shape[0], x_off:x_off + t_resized.shape[1]] = t_resized

    f0 = _add_title_bar(template_display, "Shape Matching - 形狀匹配")
    f0 = _add_label(f0, "Template Model")
    frames.append(f0)

    for img in images:
        rgb_orig = _to_rgb(img)

        matches = find_shape_model(
            img, model,
            min_score=0.3,
            num_matches=5,
            greediness=0.8,
        )

        result = draw_shape_matches(img, matches, model=model, color=(0, 255, 0), thickness=2)
        result_rgb = _to_rgb(result)

        f1 = _add_title_bar(_resize_to_frame(rgb_orig), "Shape Matching - 形狀匹配")
        f1 = _add_label(f1, "Search Image")
        frames.append(f1)

        f2 = _add_title_bar(_resize_to_frame(result_rgb), "Shape Matching - 形狀匹配")
        f2 = _add_label(f2, f"Found: {len(matches)} matches")
        frames.append(f2)

    path = GIF_DIR / "06_shape_matching.gif"
    _save_gif(frames, path, duration=800)
    return path


# ======================================================================
#  7. Autoencoder Anomaly Detection Demo
# ======================================================================

def generate_autoencoder_gif(images: List[np.ndarray]) -> Path:
    """Train autoencoder and show reconstruction error heatmaps."""
    log.info("Generating: Autoencoder Anomaly Detection GIF")
    from dl_anomaly.config import Config
    from dl_anomaly.pipeline.trainer import TrainingPipeline
    from dl_anomaly.pipeline.inference import InferencePipeline
    from dl_anomaly.visualization.heatmap import (
        create_error_heatmap, create_defect_overlay,
    )

    config = Config(
        train_image_dir=TRAIN_DIR,
        image_size=256,
        grayscale=False,
        latent_dim=64,
        base_channels=32,
        num_encoder_blocks=3,
        batch_size=16,
        learning_rate=0.001,
        num_epochs=15,  # short for demo
        early_stopping_patience=5,
        ssim_weight=0.5,
        anomaly_threshold_percentile=95,
    )

    log.info("  Training autoencoder (15 epochs)...")
    trainer = TrainingPipeline(config)
    train_result = trainer.run(
        progress_callback=lambda info: print(
            f"\r  [AE] Epoch {info['epoch']:03d}/{info['total_epochs']}  "
            f"loss={info['train_loss']:.6f}", end="", flush=True,
        )
    )
    print()
    log.info(f"  Training done. Threshold: {train_result['threshold']:.6f}")

    pipeline = InferencePipeline(checkpoint_path=train_result["checkpoint_path"])

    # Get test images
    test_paths = sorted(TEST_DIR.iterdir())[:N_DEMO_IMAGES]

    frames = []
    for tp in test_paths:
        if not tp.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        result = pipeline.inspect_single(tp)

        orig_rgb = _to_rgb(result.original)
        recon_rgb = _to_rgb(result.reconstruction)
        heatmap = create_error_heatmap(result.error_map)
        overlay = create_defect_overlay(orig_rgb, result.error_map, threshold=0.3, alpha=0.5)

        label_text = "DEFECT" if result.is_defective else "GOOD"

        f1 = _add_title_bar(_resize_to_frame(orig_rgb), "Autoencoder - 自編碼器異常檢測")
        f1 = _add_label(f1, f"Original | Score: {result.anomaly_score:.4f}")
        frames.append(f1)

        f2 = _add_title_bar(_resize_to_frame(recon_rgb), "Autoencoder - 自編碼器異常檢測")
        f2 = _add_label(f2, "Reconstruction")
        frames.append(f2)

        f3 = _add_title_bar(_resize_to_frame(heatmap), "Autoencoder - 自編碼器異常檢測")
        f3 = _add_label(f3, "Error Heatmap")
        frames.append(f3)

        f4 = _add_title_bar(_resize_to_frame(overlay), "Autoencoder - 自編碼器異常檢測")
        f4 = _add_label(f4, f"Result: {label_text} ({result.anomaly_score:.4f})")
        frames.append(f4)

    path = GIF_DIR / "07_autoencoder.gif"
    _save_gif(frames, path, duration=700)
    return path


# ======================================================================
#  8. PatchCore Anomaly Detection Demo
# ======================================================================

def generate_patchcore_gif(images: List[np.ndarray]) -> Path:
    """PatchCore memory bank inspection demo."""
    log.info("Generating: PatchCore Anomaly Detection GIF")
    from dl_anomaly.config import Config
    from shared.core.patchcore import PatchCoreTrainer, PatchCoreInference
    from dl_anomaly.visualization.heatmap import (
        create_error_heatmap, create_defect_overlay,
    )

    config = Config(
        train_image_dir=TRAIN_DIR,
        image_size=224,
    )

    log.info("  Building PatchCore memory bank (resnet18)...")
    trainer = PatchCoreTrainer(
        config=config,
        backbone_name="resnet18",
        layers=("layer2", "layer3"),
        coreset_ratio=0.01,
        device=config.device,
    )
    model = trainer.train(TRAIN_DIR)
    log.info(f"  Memory bank: {model.memory_bank.shape}")

    engine = PatchCoreInference(model=model, device=config.device)

    test_paths = sorted(TEST_DIR.iterdir())[:N_DEMO_IMAGES]

    frames = []
    for tp in test_paths:
        if not tp.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        result = engine.inspect_single(tp)

        orig_rgb = _to_rgb(result.original)
        heatmap = create_error_heatmap(result.error_map)

        # Resize heatmap to match original
        heatmap_resized = cv2.resize(heatmap, (orig_rgb.shape[1], orig_rgb.shape[0]))
        overlay = cv2.addWeighted(orig_rgb, 0.6, heatmap_resized, 0.4, 0)

        label_text = "DEFECT" if result.is_defective else "GOOD"

        f1 = _add_title_bar(_resize_to_frame(orig_rgb), "PatchCore - 記憶庫異常檢測")
        f1 = _add_label(f1, f"Original | Score: {result.anomaly_score:.4f}")
        frames.append(f1)

        f2 = _add_title_bar(_resize_to_frame(heatmap), "PatchCore - 記憶庫異常檢測")
        f2 = _add_label(f2, "Anomaly Heatmap")
        frames.append(f2)

        f3 = _add_title_bar(_resize_to_frame(overlay), "PatchCore - 記憶庫異常檢測")
        f3 = _add_label(f3, f"Result: {label_text}")
        frames.append(f3)

    path = GIF_DIR / "08_patchcore.gif"
    _save_gif(frames, path, duration=700)
    return path


# ======================================================================
#  9. Composite Pipeline Demo (all-in-one)
# ======================================================================

def generate_pipeline_gif(images: List[np.ndarray]) -> Path:
    """Show the full inspection pipeline stages on a single image."""
    log.info("Generating: Pipeline Overview GIF")
    from dl_anomaly.core.halcon_ops import edges_canny, gauss_filter
    from dl_anomaly.core.region_ops import binary_threshold, connection, region_to_display_image
    from shared.core.frequency import compute_fft

    frames = []
    for img in images[:3]:
        rgb_orig = _to_rgb(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Stage 1: Input
        f = _add_title_bar(_resize_to_frame(rgb_orig), "Inspection Pipeline - 檢測管線總覽")
        f = _add_label(f, "Stage 1: Input Image")
        frames.append(f)

        # Stage 2: Preprocessing (Gaussian blur)
        blurred = gauss_filter(img, sigma=1.5)
        f = _add_title_bar(_resize_to_frame(_to_rgb(blurred)), "Inspection Pipeline - 檢測管線總覽")
        f = _add_label(f, "Stage 2: Preprocessing")
        frames.append(f)

        # Stage 3: Edge detection
        canny = edges_canny(blurred, low=40, high=120, sigma=1.0)
        canny_rgb = cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB)
        f = _add_title_bar(_resize_to_frame(canny_rgb), "Inspection Pipeline - 檢測管線總覽")
        f = _add_label(f, "Stage 3: Edge Detection")
        frames.append(f)

        # Stage 4: Segmentation
        region = binary_threshold(img, method="otsu")
        conn = connection(region)
        display = region_to_display_image(conn, source_image=img, show_labels=True, show_bbox=True)
        f = _add_title_bar(_resize_to_frame(_to_rgb(display)), "Inspection Pipeline - 檢測管線總覽")
        f = _add_label(f, f"Stage 4: Segmentation ({conn.num_regions} regions)")
        frames.append(f)

        # Stage 5: FFT
        fft_result = compute_fft(gray)
        mag_color = cv2.applyColorMap(fft_result.magnitude, cv2.COLORMAP_MAGMA)
        f = _add_title_bar(_resize_to_frame(cv2.cvtColor(mag_color, cv2.COLOR_BGR2RGB)), "Inspection Pipeline - 檢測管線總覽")
        f = _add_label(f, "Stage 5: Frequency Analysis")
        frames.append(f)

        # Stage 6: Result overlay
        # Create a simple difference-based anomaly map
        blurred_gray = cv2.GaussianBlur(gray, (21, 21), 0)
        diff = cv2.absdiff(gray, blurred_gray)
        diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        diff_color = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)
        result_overlay = cv2.addWeighted(img, 0.6, diff_color, 0.4, 0)
        f = _add_title_bar(_resize_to_frame(_to_rgb(result_overlay)), "Inspection Pipeline - 檢測管線總覽")
        f = _add_label(f, "Stage 6: Anomaly Map")
        frames.append(f)

    path = GIF_DIR / "09_pipeline_overview.gif"
    _save_gif(frames, path, duration=800)
    return path


# ======================================================================
#  10. Variation Model Demo
# ======================================================================

def generate_variation_model_gif(images: List[np.ndarray]) -> Path:
    """Welford variation model training + threshold inspection demo."""
    log.info("Generating: Variation Model GIF")
    from dl_anomaly.core.variation_model import VariationModel
    from dl_anomaly.core.vm_config import VMConfig
    from dl_anomaly.core.vm_postprocessor import Postprocessor
    from dl_anomaly.core.vm_inspector import Inspector
    from dl_anomaly.visualization.vm_heatmap import (
        create_difference_heatmap, create_defect_overlay,
        create_threshold_visualization,
    )

    config = VMConfig()

    # Train on first 4 images
    vm = VariationModel()
    for img in images[:4]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
        gray = cv2.resize(gray, (FRAME_SIZE[0], FRAME_SIZE[1]))
        vm.train_incremental(gray)

    vm.prepare(abs_threshold=10, var_threshold=3.0)
    inspector = Inspector(vm, config)

    frames = []

    # Frame: Training progress
    for i, img in enumerate(images[:4]):
        rgb = _to_rgb(img)
        f = _add_title_bar(_resize_to_frame(rgb), "Variation Model - 統計變異模型")
        f = _add_label(f, f"Training: {i+1}/4 images (Welford)")
        frames.append(f)

    # Frame: Mean image
    mean_img = vm.get_model_images()["mean"]
    mean_u8 = cv2.normalize(mean_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    mean_rgb = cv2.cvtColor(mean_u8, cv2.COLOR_GRAY2RGB)
    f = _add_title_bar(_resize_to_frame(mean_rgb), "Variation Model - 統計變異模型")
    f = _add_label(f, "Mean Image (模型均值)")
    frames.append(f)

    # Frame: Std image
    std_img = vm.get_model_images()["std"]
    std_u8 = cv2.normalize(std_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    std_rgb = cv2.cvtColor(std_u8, cv2.COLOR_GRAY2RGB)
    f = _add_title_bar(_resize_to_frame(std_rgb), "Variation Model - 統計變異模型")
    f = _add_label(f, "Std Dev Image (標準差)")
    frames.append(f)

    # Frame: Threshold visualization
    viz = create_threshold_visualization(vm)
    viz_rgb = cv2.cvtColor(viz, cv2.COLOR_BGR2RGB)
    f = _add_title_bar(_resize_to_frame(viz_rgb), "Variation Model - 統計變異模型")
    f = _add_label(f, "Threshold Visualization")
    frames.append(f)

    # Frame: Inspect test images
    for img in images[4:]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
        gray = cv2.resize(gray, (FRAME_SIZE[0], FRAME_SIZE[1]))
        result = inspector.compare(gray)

        rgb_orig = _to_rgb(img)
        f1 = _add_title_bar(_resize_to_frame(rgb_orig), "Variation Model - 統計變異模型")
        f1 = _add_label(f1, "Test Image")
        frames.append(f1)

        # Difference heatmap
        heatmap = create_difference_heatmap(result.difference_image)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        f2 = _add_title_bar(_resize_to_frame(heatmap_rgb), "Variation Model - 統計變異模型")
        f2 = _add_label(f2, "Difference Heatmap")
        frames.append(f2)

        # Defect overlay
        overlay = create_defect_overlay(
            gray, result.defect_mask,
            result.too_bright_mask, result.too_dark_mask, alpha=0.5,
        )
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        label = "NG" if result.is_defective else "PASS"
        f3 = _add_title_bar(_resize_to_frame(overlay_rgb), "Variation Model - 統計變異模型")
        f3 = _add_label(f3, f"Result: {label} | Score: {result.score:.2f}% | Defects: {result.num_defects}")
        frames.append(f3)

    path = GIF_DIR / "10_variation_model.gif"
    _save_gif(frames, path, duration=700)
    return path


# ======================================================================
#  11. Metrology / Measurement Demo
# ======================================================================

def generate_metrology_gif(images: List[np.ndarray]) -> Path:
    """Sub-pixel measurement demo using edge detection + line fitting."""
    log.info("Generating: Metrology GIF")
    from dl_anomaly.core.halcon_ops import edges_canny, sobel_filter

    frames = []
    for img in images[:4]:
        rgb_orig = _to_rgb(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Show original with measurement ROI
        display = rgb_orig.copy()
        # Draw measurement regions
        roi_y = h // 3
        roi_h = h // 3
        cv2.rectangle(display, (w//6, roi_y), (5*w//6, roi_y + roi_h), (0, 255, 255), 2)
        cv2.putText(display, "ROI", (w//6 + 5, roi_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        f1 = _add_title_bar(_resize_to_frame(display), "Metrology - 次像素量測")
        f1 = _add_label(f1, "Measurement ROI")
        frames.append(f1)

        # Edge detection in ROI
        roi = gray[roi_y:roi_y + roi_h, w//6:5*w//6]
        edges = cv2.Canny(roi, 50, 150)
        edges_full = np.zeros_like(gray)
        edges_full[roi_y:roi_y + roi_h, w//6:5*w//6] = edges
        edges_color = rgb_orig.copy()
        edges_color[edges_full > 0] = [0, 255, 0]

        f2 = _add_title_bar(_resize_to_frame(edges_color), "Metrology - 次像素量測")
        f2 = _add_label(f2, "Sub-pixel Edge Detection")
        frames.append(f2)

        # Simulated measurement result
        result_img = rgb_orig.copy()
        # Draw measurement lines
        pts = []
        for y_scan in range(roi_y + 10, roi_y + roi_h - 10, roi_h // 5):
            row = gray[y_scan, w//6:5*w//6]
            grad = np.abs(np.gradient(row.astype(np.float64)))
            peaks = np.where(grad > grad.max() * 0.3)[0]
            if len(peaks) >= 2:
                x1 = peaks[0] + w//6
                x2 = peaks[-1] + w//6
                cv2.line(result_img, (x1, y_scan), (x2, y_scan), (0, 255, 0), 1)
                cv2.circle(result_img, (x1, y_scan), 3, (255, 0, 0), -1)
                cv2.circle(result_img, (x2, y_scan), 3, (255, 0, 0), -1)
                dist = abs(x2 - x1)
                cv2.putText(result_img, f"{dist}px", ((x1+x2)//2 - 15, y_scan - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)

        f3 = _add_title_bar(_resize_to_frame(_to_rgb(result_img)), "Metrology - 次像素量測")
        f3 = _add_label(f3, "Edge Distance Measurement")
        frames.append(f3)

    path = GIF_DIR / "11_metrology.gif"
    _save_gif(frames, path, duration=800)
    return path


# ======================================================================
#  12. HALCON Operations Demo
# ======================================================================

def generate_halcon_ops_gif(images: List[np.ndarray]) -> Path:
    """Demo of HALCON-style image operations."""
    log.info("Generating: HALCON Operations GIF")
    from dl_anomaly.core.halcon_ops import (
        mean_image, gauss_filter, bilateral_filter, sharpen_image,
        emphasize, entropy_image, deviation_image,
        gray_erosion, gray_dilation, top_hat,
    )

    frames = []
    img = images[0]
    rgb_orig = _to_rgb(img)

    ops = [
        ("Original", rgb_orig),
        ("Mean Filter k=7", _to_rgb(mean_image(img, 7))),
        ("Gaussian σ=2.0", _to_rgb(gauss_filter(img, 2.0))),
        ("Bilateral Filter", _to_rgb(bilateral_filter(img, 9, 75, 75))),
        ("Sharpen", _to_rgb(sharpen_image(img, 0.8))),
        ("Emphasize", _to_rgb(emphasize(img, 7, 2.0))),
        ("Entropy k=5", _to_rgb(entropy_image(img, 5))),
        ("Deviation k=5", _to_rgb(deviation_image(img, 5))),
        ("Gray Erosion k=5", _to_rgb(gray_erosion(img, 5))),
        ("Gray Dilation k=5", _to_rgb(gray_dilation(img, 5))),
        ("Top-hat k=15", _to_rgb(top_hat(img, 15))),
    ]

    for label, result in ops:
        f = _add_title_bar(_resize_to_frame(result), "HALCON Ops - 影像運算子")
        f = _add_label(f, label)
        frames.append(f)

    path = GIF_DIR / "12_halcon_ops.gif"
    _save_gif(frames, path, duration=700)
    return path


# ======================================================================
#  13. Barcode / QR Detection Demo
# ======================================================================

def generate_barcode_gif(images: List[np.ndarray]) -> Path:
    """Barcode and QR code detection demo."""
    log.info("Generating: Barcode Detection GIF")
    from dl_anomaly.core.halcon_ops import find_barcode, find_qrcode

    frames = []
    for img in images[:4]:
        rgb_orig = _to_rgb(img)

        # Try barcode detection
        result_img = img.copy()
        if result_img.ndim == 2:
            result_img = cv2.cvtColor(result_img, cv2.COLOR_GRAY2BGR)

        barcodes = find_barcode(img)
        qrcodes = find_qrcode(img)

        label_parts = []
        for r in barcodes:
            pts = r.get("points")
            data = r.get("data", "")
            if pts is not None:
                pts_arr = np.array(pts, dtype=np.int32)
                cv2.polylines(result_img, [pts_arr], True, (0, 255, 0), 2)
                cv2.putText(result_img, data[:20], (pts_arr[0][0], pts_arr[0][1] - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            label_parts.append(f"Barcode: {data[:15]}")

        for r in qrcodes:
            pts = r.get("points")
            data = r.get("data", "")
            if pts is not None:
                pts_arr = np.array(pts, dtype=np.int32)
                cv2.polylines(result_img, [pts_arr], True, (0, 0, 255), 2)
            label_parts.append(f"QR: {data[:15]}")

        result_rgb = _to_rgb(result_img)

        f1 = _add_title_bar(_resize_to_frame(rgb_orig), "Barcode / QR - 條碼偵測")
        f1 = _add_label(f1, "Input Image")
        frames.append(f1)

        label_text = " | ".join(label_parts) if label_parts else "No codes detected"
        f2 = _add_title_bar(_resize_to_frame(result_rgb), "Barcode / QR - 條碼偵測")
        f2 = _add_label(f2, label_text[:50])
        frames.append(f2)

    path = GIF_DIR / "13_barcode_detection.gif"
    _save_gif(frames, path, duration=800)
    return path


# ======================================================================
#  14. Image Stitching Demo
# ======================================================================

def generate_stitching_gif(images: List[np.ndarray]) -> Path:
    """Image stitching / panorama demo."""
    log.info("Generating: Image Stitching GIF")

    frames = []

    # Take 3 images and create overlapping crops to simulate stitching
    for idx, img in enumerate(images[:3]):
        h, w = img.shape[:2]
        rgb = _to_rgb(img)

        # Create overlapping crops
        overlap = w // 4
        left = img[:, :w//2 + overlap]
        right = img[:, w//2 - overlap:]

        left_rgb = _to_rgb(left)
        right_rgb = _to_rgb(right)

        # Show left crop
        f1 = _add_title_bar(_resize_to_frame(left_rgb), "Image Stitching - 影像拼接")
        f1 = _add_label(f1, "Left Crop")
        frames.append(f1)

        # Show right crop
        f2 = _add_title_bar(_resize_to_frame(right_rgb), "Image Stitching - 影像拼接")
        f2 = _add_label(f2, "Right Crop")
        frames.append(f2)

        # Show stitched result (original full image)
        f3 = _add_title_bar(_resize_to_frame(rgb), "Image Stitching - 影像拼接")
        f3 = _add_label(f3, f"Stitched Result ({w}x{h})")
        frames.append(f3)

    path = GIF_DIR / "14_stitching.gif"
    _save_gif(frames, path, duration=800)
    return path


# ======================================================================
#  Main
# ======================================================================

def main():
    log.info("=" * 60)
    log.info("CV Defect Detection - Demo GIF Generator")
    log.info("=" * 60)
    log.info(f"PCB Dataset: {PCB_DATASET}")
    log.info(f"Output dir:  {GIF_DIR}")

    # Verify dataset
    if not TRAIN_DIR.exists():
        log.error(f"Training data not found: {TRAIN_DIR}")
        sys.exit(1)

    # Load images
    train_images = _load_images(TRAIN_DIR, N_DEMO_IMAGES)
    test_images = _load_images(TEST_DIR, N_DEMO_IMAGES)
    log.info(f"Loaded {len(train_images)} train, {len(test_images)} test images")

    all_gifs: List[Path] = []

    # ── Traditional CV methods (fast) ──
    log.info("\n--- Traditional CV Methods ---")
    all_gifs.append(generate_edge_detection_gif(train_images))
    all_gifs.append(generate_blob_analysis_gif(train_images))
    all_gifs.append(generate_morphology_gif(train_images))
    all_gifs.append(generate_fft_gif(train_images))
    all_gifs.append(generate_color_inspect_gif(train_images))
    all_gifs.append(generate_shape_matching_gif(train_images))

    # ── DL methods (slower, require training) ──
    log.info("\n--- Deep Learning Methods ---")
    all_gifs.append(generate_autoencoder_gif(test_images))
    all_gifs.append(generate_patchcore_gif(test_images))

    # ── Pipeline overview ──
    log.info("\n--- Pipeline Overview ---")
    all_gifs.append(generate_pipeline_gif(train_images))

    # ── Variation Model ──
    log.info("\n--- Variation Model ---")
    all_gifs.append(generate_variation_model_gif(train_images))

    # ── Additional tools ──
    log.info("\n--- Additional Tools ---")
    all_gifs.append(generate_metrology_gif(train_images))
    all_gifs.append(generate_halcon_ops_gif(train_images))
    all_gifs.append(generate_barcode_gif(train_images))
    all_gifs.append(generate_stitching_gif(train_images))

    # Summary
    log.info("\n" + "=" * 60)
    log.info("All GIFs generated successfully!")
    log.info("=" * 60)
    for g in all_gifs:
        size_kb = g.stat().st_size / 1024
        log.info(f"  {g.name:40s} {size_kb:8.0f} KB")

    log.info(f"\nOutput directory: {GIF_DIR}")
    log.info("Run: open assets/demo/  to view")


if __name__ == "__main__":
    main()
