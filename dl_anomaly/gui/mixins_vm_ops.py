"""Variation Model operations mixin for InspectorApp.

Provides VM-specific training, inspection, model management, and
threshold recalculation capabilities integrated into the unified app.
"""

from __future__ import annotations

import logging
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import TYPE_CHECKING, Dict, Optional

import cv2
import numpy as np

if TYPE_CHECKING:
    from dl_anomaly.gui.inspector_app import InspectorApp

logger = logging.getLogger(__name__)


class VMOpsMixin:
    """Variation Model training, inspection, and model management."""

    # ==================================================================
    # VM helper: centralised VMConfig access
    # ==================================================================

    def _get_vm_config(self: "InspectorApp") -> "VMConfig":
        """Return a VMConfig loaded from .env (cached per call)."""
        from dl_anomaly.core.vm_config import VMConfig
        return VMConfig.from_env()

    # ==================================================================
    # VM Model commands
    # ==================================================================

    def _cmd_vm_train(self: "InspectorApp") -> None:
        """Train a new Variation Model from a directory of good images."""
        d = filedialog.askdirectory(title="選擇訓練影像目錄 (良品)")
        if not d:
            return

        vm_config = self._get_vm_config().update(train_image_dir=d)
        train_dir = Path(d)

        def _progress_cb(current: int, total: int) -> None:
            self.after(0, lambda c=current, t=total: self.set_status(
                f"VM 訓練中... ({c}/{t})"
            ))

        def _compute():
            from dl_anomaly.pipeline.vm_trainer import TrainingPipeline
            pipeline = TrainingPipeline(vm_config)
            try:
                model = pipeline.run(
                    image_dir=train_dir, progress_callback=_progress_cb,
                )
            except TypeError:
                # Fallback if pipeline.run does not accept progress_callback
                model = pipeline.run(image_dir=train_dir)
            return model

        def _done(model):
            self._vm_model = model
            self.set_status(f"VM 訓練完成: {model.count} 張影像")

            # Add model images to pipeline
            imgs = model.get_model_images()
            if imgs["mean"] is not None:
                mean_u8 = self._vm_to_display(imgs["mean"])
                self._add_pipeline_step("VM 模型均值", mean_u8)
            if imgs["std"] is not None:
                std_u8 = self._vm_to_display(imgs["std"])
                self._add_pipeline_step("VM 模型標準差", std_u8)

            messagebox.showinfo(
                "訓練完成",
                f"Variation Model 訓練完成。\n"
                f"訓練影像數: {model.count}",
            )

        def _error(exc):
            self._show_error("VM 訓練失敗", exc)

        self._run_in_bg(_compute, on_done=_done, on_error=_error,
                        status_msg="VM 訓練中...")

    def _cmd_vm_load_model(self: "InspectorApp") -> None:
        """Load a saved Variation Model (.npz)."""
        path = filedialog.askopenfilename(
            title="載入 Variation Model",
            filetypes=[("NumPy Archive", "*.npz"), ("所有檔案", "*")],
        )
        if not path:
            return

        try:
            from dl_anomaly.core.variation_model import VariationModel
            model = VariationModel.load(path)
            self._vm_model = model

            # Ensure thresholds are prepared using VMConfig defaults
            if model.get_model_images()["upper"] is None:
                cfg = self._get_vm_config()
                model.prepare(abs_threshold=cfg.abs_threshold, var_threshold=cfg.var_threshold)

            self.set_status(f"VM 模型已載入: {Path(path).name} ({model.count} 張影像)")
            messagebox.showinfo("載入成功", f"Variation Model 已載入。\n訓練影像數: {model.count}")
        except Exception as exc:
            self._show_error("VM 模型載入失敗", exc)

    def _cmd_vm_save_model(self: "InspectorApp") -> None:
        """Save the current Variation Model to .npz."""
        if self._vm_model is None or not self._vm_model.is_trained:
            messagebox.showinfo("資訊", "沒有已訓練的 Variation Model 可以儲存。")
            return

        path = filedialog.asksaveasfilename(
            title="儲存 Variation Model",
            defaultextension=".npz",
            filetypes=[("NumPy Archive", "*.npz")],
        )
        if not path:
            return

        try:
            self._vm_model.save(path)
            self.set_status(f"VM 模型已儲存: {Path(path).name}")
        except Exception as exc:
            self._show_error("VM 模型儲存失敗", exc)

    def _cmd_vm_model_info(self: "InspectorApp") -> None:
        """Show Variation Model information."""
        if self._vm_model is None:
            messagebox.showinfo("資訊", "尚未載入 Variation Model。")
            return

        model = self._vm_model
        imgs = model.get_model_images()
        info = (
            f"Variation Model 資訊\n\n"
            f"訓練影像數: {model.count}\n"
            f"模型狀態: {'已訓練' if model.is_trained else '未訓練'}\n"
            f"絕對閾值: {model._abs_threshold}\n"
            f"變異閾值: {model._var_threshold}\n"
            f"均值影像: {'有' if imgs['mean'] is not None else '無'}\n"
            f"標準差影像: {'有' if imgs['std'] is not None else '無'}\n"
            f"上界閾值: {'有' if imgs['upper'] is not None else '無'}\n"
            f"下界閾值: {'有' if imgs['lower'] is not None else '無'}\n"
        )
        if imgs["mean"] is not None:
            info += f"影像尺寸: {imgs['mean'].shape}\n"

        messagebox.showinfo("VM 模型資訊", info)

    def _cmd_vm_reprepare_thresholds(self: "InspectorApp") -> None:
        """Recalculate VM thresholds with current parameters."""
        if self._vm_model is None or not self._vm_model.is_trained:
            messagebox.showinfo("資訊", "沒有已訓練的 Variation Model。")
            return

        try:
            cfg = self._get_vm_config()
            self._vm_model.prepare(abs_threshold=cfg.abs_threshold, var_threshold=cfg.var_threshold)
            self.set_status("VM 閾值已重新計算")
            messagebox.showinfo("完成", "Variation Model 閾值已重新計算。")
        except Exception as exc:
            self._show_error("VM 閾值計算失敗", exc)

    # ==================================================================
    # VM Inspection commands
    # ==================================================================

    def _cmd_vm_inspect_single(self: "InspectorApp") -> None:
        """Run VM inspection on the current or selected image."""
        if self._vm_model is None or not self._vm_model.is_trained:
            messagebox.showinfo("資訊", "請先載入或訓練 Variation Model。")
            return

        # Ensure thresholds -- confirm with user before auto-preparing
        imgs = self._vm_model.get_model_images()
        if imgs["upper"] is None:
            cfg = self._get_vm_config()
            if not messagebox.askyesno(
                "閾值未設定",
                f"模型尚未計算檢測閾值。\n"
                f"是否使用預設參數自動計算？\n\n"
                f"  絕對閾值: {cfg.abs_threshold}\n"
                f"  變異閾值: {cfg.var_threshold}",
            ):
                return
            try:
                self._vm_model.prepare(
                    abs_threshold=cfg.abs_threshold,
                    var_threshold=cfg.var_threshold,
                )
            except Exception as exc:
                self._show_error("VM 閾值計算失敗", exc)
                return

        path = self._current_image_path
        if path is None or not Path(path).exists():
            path = filedialog.askopenfilename(
                title="選擇要檢測的影像",
                filetypes=[("影像", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")],
            )
            if not path:
                return

        vm_model = self._vm_model
        inspect_path = Path(path)

        vm_config = self._get_vm_config()

        def _compute():
            from dl_anomaly.pipeline.vm_inference import InferencePipeline
            from dl_anomaly.visualization.vm_heatmap import (
                create_defect_overlay,
                create_difference_heatmap,
            )

            pipeline = InferencePipeline(vm_model, vm_config)
            result, processed = pipeline.inspect_single(inspect_path)

            overlay = create_defect_overlay(
                processed, result.defect_mask,
                result.too_bright_mask, result.too_dark_mask, alpha=0.5,
            )
            heatmap = create_difference_heatmap(result.difference_image)
            return result, processed, overlay, heatmap

        def _done(payload):
            result, processed, overlay, heatmap = payload
            self._vm_display_result(result, processed, overlay, heatmap, inspect_path)

        def _error(exc):
            self._show_error("VM 檢測失敗", exc)

        self._run_in_bg(_compute, on_done=_done, on_error=_error,
                        status_msg=f"VM 檢測中: {inspect_path.name}")

    def _vm_display_result(
        self: "InspectorApp", result, processed, overlay, heatmap, image_path
    ) -> None:
        """Display VM inspection results in the pipeline."""
        name = image_path.stem

        # Convert images for display
        processed_disp = self._vm_to_display(processed)
        heatmap_disp = self._vm_ensure_rgb(heatmap)
        overlay_disp = self._vm_ensure_rgb(overlay)
        mask_disp = self._vm_to_display(result.defect_mask)

        self._add_pipeline_step(f"VM 前處理: {name}", processed_disp)
        self._add_pipeline_step("VM 差異熱力圖", heatmap_disp)
        self._add_pipeline_step("VM 瑕疵遮罩", mask_disp)
        self._add_pipeline_step("VM 瑕疵疊加", overlay_disp)

        status = "NG - 瑕疵" if result.is_defective else "PASS - 合格"
        self.set_status(
            f"VM 檢測完成: {image_path.name} - {status}  |  "
            f"分數: {result.score:.4f}%  |  瑕疵數: {result.num_defects}"
        )

        # Update OK/NG judgment indicator
        self.update_judgment(
            is_pass=not result.is_defective,
            score=result.score,
            message=f"VM 瑕疵數: {result.num_defects}",
        )

    def _cmd_vm_inspect_batch(self: "InspectorApp") -> None:
        """Run VM batch inspection on a directory."""
        if self._vm_model is None or not self._vm_model.is_trained:
            messagebox.showinfo("資訊", "請先載入或訓練 Variation Model。")
            return

        # Ensure thresholds
        imgs = self._vm_model.get_model_images()
        if imgs["upper"] is None:
            cfg = self._get_vm_config()
            try:
                self._vm_model.prepare(
                    abs_threshold=cfg.abs_threshold,
                    var_threshold=cfg.var_threshold,
                )
            except Exception as exc:
                self._show_error("VM 閾值計算失敗", exc)
                return

        d = filedialog.askdirectory(title="選擇測試影像目錄")
        if not d:
            return

        out_dir = filedialog.askdirectory(title="選擇結果輸出目錄")
        if not out_dir:
            return

        vm_model = self._vm_model
        test_dir = d
        output_dir = out_dir
        vm_config = self._get_vm_config()

        def _compute():
            from dl_anomaly.pipeline.vm_inference import InferencePipeline

            pipeline = InferencePipeline(vm_model, vm_config)
            results = pipeline.inspect_batch(test_dir)
            report_dir = pipeline.generate_report(results, output_dir)
            return len(results), report_dir

        def _done(payload):
            count, report_dir = payload
            self.set_status(f"VM 批次檢測完成: {count} 張影像")
            messagebox.showinfo(
                "批次檢測完成",
                f"已檢測 {count} 張影像。\n報告輸出至: {report_dir}",
            )

        def _error(exc):
            self._show_error("VM 批次檢測失敗", exc)

        self._run_in_bg(_compute, on_done=_done, on_error=_error,
                        status_msg="VM 批次檢測中...")

    def _cmd_vm_show_threshold_viz(self: "InspectorApp") -> None:
        """Show the VM model's threshold visualization (mean/std/upper/lower)."""
        if self._vm_model is None or not self._vm_model.is_trained:
            messagebox.showinfo("資訊", "請先載入或訓練 Variation Model。")
            return

        try:
            from dl_anomaly.visualization.vm_heatmap import create_threshold_visualization
            grid = create_threshold_visualization(self._vm_model)
            grid_rgb = self._vm_ensure_rgb(grid)
            self._add_pipeline_step("VM 閾值視覺化", grid_rgb)
            self.set_status("VM 閾值視覺化已生成")
        except Exception as exc:
            self._show_error("VM 閾值視覺化失敗", exc)

    # ==================================================================
    # VM helper methods
    # ==================================================================

    @staticmethod
    def _vm_to_display(image: np.ndarray) -> np.ndarray:
        """Convert a VM image (possibly float64) to uint8 RGB for display."""
        if image.dtype == np.uint8:
            arr = image
        else:
            min_val = image.min()
            max_val = image.max()
            if max_val - min_val > 0:
                arr = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            else:
                arr = np.zeros_like(image, dtype=np.uint8)

        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        elif arr.shape[2] == 3:
            # VM uses BGR (OpenCV), convert to RGB for PIL display
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        return arr

    @staticmethod
    def _vm_ensure_rgb(image: np.ndarray) -> np.ndarray:
        """Ensure image is RGB uint8 for display."""
        if image.dtype != np.uint8:
            min_val = image.min()
            max_val = image.max()
            if max_val - min_val > 0:
                image = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            else:
                image = np.zeros_like(image, dtype=np.uint8)

        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if image.ndim == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
