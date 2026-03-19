"""Image operations mixin for InspectorApp.

Covers: file I/O, undo/redo, view commands, pipeline helpers,
model commands, inspection commands, DL operations, recipe commands,
help commands, and settings.
"""

from __future__ import annotations

import logging
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Any, Dict, Optional, TYPE_CHECKING

import cv2

from dl_anomaly.gui.platform_keys import display, display_shift, DELETE_LABEL
import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from dl_anomaly.gui.inspector_app import InspectorApp

logger = logging.getLogger(__name__)


class ImageOpsMixin:
    """File I/O, pipeline management, model/inspection, DL ops, recipes, help."""

    # ==================================================================
    # Pipeline helpers
    # ==================================================================

    def _add_pipeline_step(self: "InspectorApp", name: str, array: np.ndarray, op_meta=None) -> None:
        """Add a new step to the pipeline and display it."""
        ci = self._pipeline_panel.get_current_index()
        if ci >= 0:
            self._undo_stack.append(ci)
        self._redo_stack.clear()
        self._pipeline_panel.add_step(name, array, select=True, op_meta=op_meta)

    def _load_image_to_pipeline(self: "InspectorApp", path: str) -> None:
        """Load an image file and add it as the first pipeline step."""
        from PIL import Image as PILImage

        img = PILImage.open(path)
        MAX_SIDE = 1080
        w, h = img.size
        if max(w, h) > MAX_SIDE:
            scale = MAX_SIDE / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), PILImage.LANCZOS)
        if img.mode == "L":
            array = np.array(img)
        elif img.mode in ("RGB", "RGBA"):
            array = np.array(img.convert("RGB"))
        else:
            array = np.array(img.convert("RGB"))

        self._current_image_path = path
        name = Path(path).stem
        op_meta = {"category": "source"}
        if not self._initial_loaded:
            op_meta["initial"] = True
        self._pipeline_panel.add_step(
            f"\u539f\u5716: {name}", array, select=True,
            op_meta=op_meta,
        )
        if not self._initial_loaded:
            self._initial_loaded = True
        self.set_status(f"\u5df2\u8f09\u5165: {Path(path).name}")

        if path not in self._recent_files:
            self._recent_files.insert(0, path)
            self._recent_files = self._recent_files[:10]
        self._app_state.add_recent_file(path)
        self._update_recent_menu()

    # ==================================================================
    # File commands
    # ==================================================================

    def _cmd_open_image(self: "InspectorApp") -> None:
        path = filedialog.askopenfilename(
            title="\u958b\u555f\u5716\u7247",
            filetypes=[
                ("\u5716\u7247\u6a94\u6848", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("\u6240\u6709\u6a94\u6848", "*"),
            ],
        )
        if path:
            try:
                self._load_image_to_pipeline(path)
            except Exception as exc:
                self._show_error("\u7121\u6cd5\u8f09\u5165\u5716\u7247", exc)

    def _cmd_open_dir(self: "InspectorApp") -> None:
        d = filedialog.askdirectory(title="\u958b\u555f\u5716\u7247\u8cc7\u6599\u593e")
        if not d:
            return
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        paths = sorted(str(p) for p in Path(d).rglob("*") if p.is_file() and p.suffix.lower() in exts)
        if not paths:
            messagebox.showinfo("\u8cc7\u8a0a", "\u8cc7\u6599\u593e\u4e2d\u6c92\u6709\u627e\u5230\u5716\u7247\u6a94\u6848")
            return
        self._load_all_images_to_pipeline(paths)

    def _load_all_images_to_pipeline(self: "InspectorApp", paths: list) -> None:
        """Load all images from *paths* as sequential pipeline steps."""
        from PIL import Image as PILImage

        self._current_image_path = paths[0]
        mark_initial = not self._initial_loaded

        MAX_SIDE = 1080
        for i, path in enumerate(paths):
            try:
                img = PILImage.open(path)
                w, h = img.size
                if max(w, h) > MAX_SIDE:
                    scale = MAX_SIDE / max(w, h)
                    new_w, new_h = int(w * scale), int(h * scale)
                    img = img.resize((new_w, new_h), PILImage.LANCZOS)
                if img.mode == "L":
                    array = np.array(img)
                elif img.mode in ("RGB", "RGBA"):
                    array = np.array(img.convert("RGB"))
                else:
                    array = np.array(img.convert("RGB"))
                name = Path(path).stem
                op_meta = {"category": "source"}
                if mark_initial:
                    op_meta["initial"] = True
                self._pipeline_panel.add_step(
                    f"\u539f\u5716: {name}", array, select=(i == 0),
                    op_meta=op_meta,
                )
            except Exception as exc:
                logger.warning("Failed to load %s: %s", path, exc)

        if mark_initial:
            self._initial_loaded = True
        self.set_status(f"\u5df2\u8f09\u5165 {len(paths)} \u5f35\u5716\u7247")

    def _cmd_save_image(self: "InspectorApp") -> None:
        step = self._pipeline_panel.get_current_step()
        if step is None:
            messagebox.showinfo("\u8cc7\u8a0a", "\u6c92\u6709\u53ef\u5132\u5b58\u7684\u5716\u7247")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp")],
            initialfile=f"{step.name}.png",
        )
        if path:
            if step.array.ndim == 2:
                img = Image.fromarray(step.array, mode="L")
            else:
                img = Image.fromarray(step.array, mode="RGB")
            img.save(path)
            self.set_status(f"\u5df2\u5132\u5b58: {path}")

    def _cmd_save_all(self: "InspectorApp") -> None:
        steps = self._pipeline_panel.get_all_steps()
        if not steps:
            messagebox.showinfo("\u8cc7\u8a0a", "\u6c92\u6709\u53ef\u5132\u5b58\u7684\u6b65\u9a5f")
            return
        d = filedialog.askdirectory(title="\u9078\u64c7\u8f38\u51fa\u8cc7\u6599\u593e")
        if not d:
            return
        for i, step in enumerate(steps):
            fname = f"{i + 1:02d}_{step.name}.png"
            fname = "".join(c if c.isalnum() or c in "._- " else "_" for c in fname)
            out_path = Path(d) / fname
            if step.array.ndim == 2:
                img = Image.fromarray(step.array, mode="L")
            else:
                img = Image.fromarray(step.array, mode="RGB")
            img.save(str(out_path))
        self.set_status(f"\u5df2\u5132\u5b58 {len(steps)} \u500b\u6b65\u9a5f\u81f3 {d}")

    # ==================================================================
    # Undo / Redo
    # ==================================================================

    def _cmd_undo(self: "InspectorApp") -> None:
        if not self._undo_stack:
            return
        current = self._pipeline_panel.get_current_index()
        self._redo_stack.append(current)
        target = self._undo_stack.pop()
        self._pipeline_panel.select_step(target)

    def _cmd_redo(self: "InspectorApp") -> None:
        if not self._redo_stack:
            return
        current = self._pipeline_panel.get_current_index()
        self._undo_stack.append(current)
        target = self._redo_stack.pop()
        self._pipeline_panel.select_step(target)

    # ==================================================================
    # View commands
    # ==================================================================

    def _cmd_fit(self: "InspectorApp") -> None:
        self._viewer.fit_to_window()

    def _cmd_zoom_in(self: "InspectorApp") -> None:
        self._viewer.zoom_in()

    def _cmd_zoom_out(self: "InspectorApp") -> None:
        self._viewer.zoom_out()

    def _cmd_actual_size(self: "InspectorApp") -> None:
        self._viewer.zoom_to_actual()

    def _cmd_toggle_grid(self: "InspectorApp", state: bool = False) -> None:
        self._viewer.set_grid(state)

    def _cmd_toggle_crosshair(self: "InspectorApp", state: bool = False) -> None:
        self._viewer.set_crosshair(state)

    def _cmd_histogram(self: "InspectorApp") -> None:
        from dl_anomaly.gui.dialogs import HistogramDialog

        step = self._pipeline_panel.get_current_step()
        if step is None:
            messagebox.showinfo("\u8cc7\u8a0a", "\u8acb\u5148\u8f09\u5165\u5716\u7247")
            return
        HistogramDialog(self, step.array, title_text=f"\u76f4\u65b9\u5716 - {step.name}")

    def _cmd_toggle_loss_curve(self: "InspectorApp") -> None:
        if self._viewer._loss_panel_visible:
            self._viewer.hide_loss_panel()
        else:
            self._viewer.show_loss_panel()

    def _cmd_reconstruction_compare(self: "InspectorApp") -> None:
        from dl_anomaly.gui.dialogs import ReconstructionDialog

        steps = self._pipeline_panel.get_all_steps()
        original = None
        reconstruction = None
        for s in steps:
            if "\u539f\u5716" in s.name or "Original" in s.name:
                original = s.array
            if "\u91cd\u5efa" in s.name or "Reconstruct" in s.name:
                reconstruction = s.array
        if original is None or reconstruction is None:
            messagebox.showinfo("\u8cc7\u8a0a", "\u9700\u8981\u539f\u5716\u548c\u91cd\u5efa\u5716\u624d\u80fd\u5c0d\u6bd4")
            return
        ReconstructionDialog(self, original, reconstruction)

    def _cmd_compare_steps(self: "InspectorApp") -> None:
        """Open a subtraction-based image comparison dialog with rules."""
        from dl_anomaly.gui.compare_dialog import CompareDialog

        items = self._collect_compare_items()
        if len(items) < 2:
            messagebox.showinfo("\u8cc7\u8a0a", "\u9700\u8981\u81f3\u5c11\u5169\u5f35\u53ef\u6bd4\u5c0d\u7684\u5716\u7247")
            return
        CompareDialog(self, items, fetch_steps_cb=self._collect_compare_items)

    def _collect_compare_items(self: "InspectorApp"):
        """Return all pipeline steps as (index, name, array) list."""
        steps = self._pipeline_panel.get_all_steps()
        return [(i, s.name, s.array) for i, s in enumerate(steps)]

    def _cmd_batch_compare_steps(self: "InspectorApp") -> None:
        """Open the batch 1-to-N image comparison dialog."""
        from dl_anomaly.gui.batch_compare_dialog import BatchCompareDialog

        items = self._collect_compare_items()
        if len(items) < 2:
            messagebox.showinfo("\u8cc7\u8a0a", "\u9700\u8981\u81f3\u5c11\u5169\u5f35\u53ef\u6bd4\u5c0d\u7684\u5716\u7247")
            return
        BatchCompareDialog(self, items,
                           fetch_steps_cb=self._collect_compare_items)

    def _cmd_delete_step(self: "InspectorApp") -> None:
        idx = self._pipeline_panel.get_current_index()
        if idx >= 0:
            self._on_pipeline_step_delete(idx)

    # ==================================================================
    # Model commands
    # ==================================================================

    def _cmd_train(self: "InspectorApp") -> None:
        from dl_anomaly.gui.dialogs import TrainingDialog

        def on_complete(result: Dict[str, Any]) -> None:
            self.set_status(
                f"\u8a13\u7df4\u5b8c\u6210  best_val={result['best_val_loss']:.6f}  "
                f"threshold={result['threshold']:.6f}"
            )
            pipeline = dlg.get_pipeline()
            if pipeline:
                history = result.get("history", {})
                tl = history.get("train_loss", [])
                vl = history.get("val_loss", [])
                self._viewer.update_loss_plot(tl, vl, title="\u8a13\u7df4\u640d\u5931\u66f2\u7dda")
                self._viewer.show_loss_panel()
            ckpt = result.get("checkpoint_path")
            if ckpt and Path(ckpt).exists():
                self._load_inference_pipeline(ckpt)

        dlg = TrainingDialog(self, self.config, on_complete=on_complete)

    def _cmd_load_model(self: "InspectorApp") -> None:
        path = filedialog.askopenfilename(
            title="\u8f09\u5165 Checkpoint",
            filetypes=[("PyTorch Checkpoint", "*.pt")],
            initialdir=str(self.config.checkpoint_dir),
        )
        if path:
            self._load_inference_pipeline(path)

    def _load_inference_pipeline(self: "InspectorApp", path: str) -> None:
        try:
            from dl_anomaly.pipeline.inference import InferencePipeline
            self._inference_pipeline = InferencePipeline(path, device=self.config.device)

            from dl_anomaly.pipeline.trainer import TrainingPipeline
            self._model, _, self._model_state = TrainingPipeline.load_checkpoint(
                Path(path), self.config.device
            )

            self.set_status(f"\u6a21\u578b\u5df2\u8f09\u5165: {Path(path).name}")
        except Exception as exc:
            self._show_error("\u6a21\u578b\u8f09\u5165\u5931\u6557", exc)

    def _cmd_save_checkpoint(self: "InspectorApp") -> None:
        if self._model is None:
            messagebox.showinfo("\u8cc7\u8a0a", "\u6c92\u6709\u53ef\u5132\u5b58\u7684\u6a21\u578b")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".pt",
            filetypes=[("PyTorch Checkpoint", "*.pt")],
            initialdir=str(self.config.checkpoint_dir),
        )
        if not path:
            return
        import torch
        state = {
            "model_state_dict": self._model.state_dict(),
            "config": self.config.to_dict(),
            "epoch": self._model_state.get("epoch", 0) if self._model_state else 0,
            "loss": self._model_state.get("loss", 0) if self._model_state else 0,
        }
        if self._model_state and "threshold" in self._model_state:
            state["threshold"] = self._model_state["threshold"]
        torch.save(state, path)
        self.set_status(f"Checkpoint \u5df2\u5132\u5b58: {path}")

    def _cmd_model_info(self: "InspectorApp") -> None:
        from dl_anomaly.gui.dialogs import ModelInfoDialog

        if self._model is None:
            messagebox.showinfo("\u8cc7\u8a0a", "\u8acb\u5148\u8f09\u5165\u6a21\u578b")
            return
        ModelInfoDialog(self, self._model, self.config, self._model_state)

    def _cmd_compute_threshold(self: "InspectorApp") -> None:
        if self._inference_pipeline is None:
            messagebox.showinfo("\u8cc7\u8a0a", "\u8acb\u5148\u8f09\u5165\u6a21\u578b")
            return
        d = filedialog.askdirectory(title="\u9078\u64c7\u8a13\u7df4\u5716\u7247\u8cc7\u6599\u593e (\u7528\u65bc\u8a08\u7b97\u95be\u503c)")
        if not d:
            return

        scorer = self._inference_pipeline.scorer
        preprocessor = self._inference_pipeline.preprocessor
        model = self._inference_pipeline.model
        device = self._inference_pipeline.device
        ssim_w = self.config.ssim_weight
        pct = self._ops_panel.get_threshold_percentile()

        def _compute():
            import torch
            exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
            paths = sorted(p for p in Path(d).rglob("*") if p.is_file() and p.suffix.lower() in exts)
            scores = []
            for p in paths:
                tensor = preprocessor.load_and_preprocess(p)
                batch = tensor.unsqueeze(0).to(device)
                with torch.no_grad():
                    recon = model(batch)
                orig_np = preprocessor.inverse_normalize(tensor)
                recon_np = preprocessor.inverse_normalize(recon.squeeze(0))
                err = scorer.compute_combined_error(orig_np, recon_np, ssim_w)
                scores.append(scorer.compute_image_score(err))
            threshold = scorer.fit_threshold(scores, pct)
            return threshold, pct, len(scores)

        def _done(result):
            threshold, pct, n = result
            self.set_status(f"\u95be\u503c\u5df2\u8a2d\u5b9a: {threshold:.6f} (percentile={pct})")
            messagebox.showinfo("\u95be\u503c", f"\u7570\u5e38\u95be\u503c: {threshold:.6f}\n(percentile={pct}, n={n})")

        self._run_in_bg(_compute, on_done=_done, status_msg="\u8a08\u7b97\u95be\u503c\u4e2d...")

    # ==================================================================
    # Inspection commands
    # ==================================================================

    def _cmd_inspect_single(self: "InspectorApp") -> None:
        """Run inspection on a single image (or the currently loaded one)."""
        if self._inference_pipeline is None:
            messagebox.showinfo("\u8cc7\u8a0a", "\u8acb\u5148\u8f09\u5165\u6a21\u578b (Checkpoint)")
            return

        path = self._current_image_path
        if path is None or not Path(path).exists():
            path = filedialog.askopenfilename(
                title="\u9078\u64c7\u8981\u6aa2\u6e2c\u7684\u5716\u7247",
                filetypes=[("\u5716\u7247", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")],
            )
            if not path:
                return

        pipeline = self._inference_pipeline
        inspect_path = path

        def _compute():
            return pipeline.inspect_single(inspect_path)

        def _done(result):
            self._display_inspection_result(result, inspect_path)
            self.set_status(f"\u6aa2\u6e2c\u5b8c\u6210: {Path(inspect_path).name}")

        def _error(exc):
            self._show_error("\u6aa2\u6e2c\u5931\u6557", exc)
            self.set_status("\u5c31\u7dd2")

        self._run_in_bg(_compute, on_done=_done, on_error=_error,
                        status_msg=f"\u6aa2\u6e2c\u4e2d: {Path(path).name}")

    def _display_inspection_result(self: "InspectorApp", result, path: str) -> None:
        """Populate the pipeline with all inspection steps."""
        from dl_anomaly.pipeline.inference import InspectionResult
        from dl_anomaly.core.anomaly_scorer import AnomalyScorer
        from dl_anomaly.visualization.heatmap import create_defect_overlay, create_error_heatmap

        name = Path(path).stem
        self._pipeline_panel.clear_all()
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._current_image_path = path

        orig = result.original
        if orig.ndim == 2:
            orig_rgb = cv2.cvtColor(orig, cv2.COLOR_GRAY2RGB)
        else:
            orig_rgb = orig

        recon = result.reconstruction
        if recon.ndim == 2:
            recon_rgb = cv2.cvtColor(recon, cv2.COLOR_GRAY2RGB)
        else:
            recon_rgb = recon

        self._pipeline_panel.add_step(f"\u539f\u5716: {name}", orig_rgb, select=False)
        self._pipeline_panel.add_step("\u524d\u8655\u7406", orig_rgb, select=False)
        self._pipeline_panel.add_step("\u91cd\u5efa", recon_rgb, select=False)

        heatmap = create_error_heatmap(result.error_map)
        self._pipeline_panel.add_step("\u8aa4\u5dee\u5716", heatmap, select=False)

        sigma = self._ops_panel.get_sigma()
        scorer = AnomalyScorer(device=self.config.device)
        smoothed = scorer.create_anomaly_map(result.error_map, gaussian_sigma=sigma)
        smoothed_heatmap = create_error_heatmap(smoothed)
        self._pipeline_panel.add_step("\u5e73\u6ed1\u8aa4\u5dee", smoothed_heatmap, select=False)

        mask_rgb = cv2.cvtColor(result.defect_mask, cv2.COLOR_GRAY2RGB)
        self._pipeline_panel.add_step("\u7f3a\u9677\u906e\u7f69", mask_rgb, select=False)

        overlay = create_defect_overlay(orig_rgb, smoothed, threshold=0.5, alpha=0.4)
        self._pipeline_panel.add_step("\u7d50\u679c\u758a\u52a0", overlay, select=True)

        rects = []
        for reg in result.defect_regions:
            rects.append({"bbox": reg["bbox"], "color": "#ff3333"})
        self._viewer.set_overlay_rects(rects)

        label = "\u7f3a\u9677" if result.is_defective else "\u901a\u904e"
        self.set_status(
            f"\u6aa2\u6e2c\u7d50\u679c: {label}  |  \u5206\u6578: {result.anomaly_score:.6f}  |  "
            f"\u7f3a\u9677\u5340\u57df: {len(result.defect_regions)}"
        )

        # Update OK/NG judgment indicator
        self.update_judgment(
            is_pass=not result.is_defective,
            score=result.anomaly_score,
            message=f"缺陷區域: {len(result.defect_regions)}",
        )

    def _cmd_batch_inspect(self: "InspectorApp") -> None:
        from dl_anomaly.gui.dialogs import BatchInspectDialog

        if self._inference_pipeline is None:
            messagebox.showinfo("\u8cc7\u8a0a", "\u8acb\u5148\u8f09\u5165\u6a21\u578b")
            return

        def on_result_selected(idx: int, result, path: str) -> None:
            self._display_inspection_result(result, path)

        BatchInspectDialog(
            self,
            self._inference_pipeline,
            on_result_selected=on_result_selected,
        )

    # ==================================================================
    # DL-specific operations (from menu)
    # ==================================================================

    def _cmd_run_autoencoder(self: "InspectorApp") -> None:
        """Feed current image through the autoencoder and add reconstruction as a step."""
        if self._inference_pipeline is None:
            messagebox.showinfo("\u8cc7\u8a0a", "\u8acb\u5148\u8f09\u5165\u6a21\u578b")
            return
        step = self._pipeline_panel.get_current_step()
        if step is None:
            messagebox.showinfo("\u8cc7\u8a0a", "\u8acb\u5148\u8f09\u5165\u5716\u7247")
            return

        arr = step.array.copy()
        preprocessor = self._inference_pipeline.preprocessor
        model = self._inference_pipeline.model
        device = self._inference_pipeline.device

        def _compute():
            import torch
            from PIL import Image as PILImage
            if arr.ndim == 2:
                pil_img = PILImage.fromarray(arr, mode="L")
            else:
                pil_img = PILImage.fromarray(arr, mode="RGB")
            transform = preprocessor.get_transforms(augment=False)
            tensor = transform(pil_img).unsqueeze(0).to(device)
            with torch.no_grad():
                recon_tensor = model(tensor)
            recon_np = preprocessor.inverse_normalize(recon_tensor.squeeze(0))
            if recon_np.ndim == 2:
                recon_np = cv2.cvtColor(recon_np, cv2.COLOR_GRAY2RGB)
            return recon_np

        def _done(recon_np):
            self._add_pipeline_step("\u91cd\u5efa", recon_np)
            self.set_status("\u81ea\u52d5\u7de8\u78bc\u5668\u57f7\u884c\u5b8c\u6210")

        self._run_in_bg(_compute, on_done=_done, status_msg="\u57f7\u884c\u81ea\u52d5\u7de8\u78bc\u5668...")

    def _cmd_compute_error_map(self: "InspectorApp") -> None:
        """Compute pixel-wise error between 'Original' and 'Reconstruction' steps."""
        steps = self._pipeline_panel.get_all_steps()
        original = None
        reconstruction = None
        for s in steps:
            if "\u539f\u5716" in s.name or "Original" in s.name:
                original = s.array
            if "\u91cd\u5efa" in s.name or "Reconstruct" in s.name:
                reconstruction = s.array

        if original is None or reconstruction is None:
            messagebox.showinfo("\u8cc7\u8a0a", "\u9700\u8981\u539f\u5716\u548c\u91cd\u5efa\u5716\u624d\u80fd\u8a08\u7b97\u8aa4\u5dee\u5716")
            return

        orig_copy = original.copy()
        recon_copy = reconstruction.copy()
        ssim_w = self._ops_panel.get_ssim_weight()
        dev = self.config.device

        def _compute():
            from dl_anomaly.core.anomaly_scorer import AnomalyScorer
            from dl_anomaly.visualization.heatmap import create_error_heatmap

            scorer = AnomalyScorer(device=dev)
            error_map = scorer.compute_combined_error(orig_copy, recon_copy, ssim_w)
            return create_error_heatmap(error_map)

        def _done(heatmap):
            self._add_pipeline_step("\u8aa4\u5dee\u5716", heatmap)
            self.set_status("\u8aa4\u5dee\u5716\u8a08\u7b97\u5b8c\u6210")

        self._run_in_bg(_compute, on_done=_done, status_msg="\u8a08\u7b97\u8aa4\u5dee\u5716...")

    def _cmd_apply_ssim(self: "InspectorApp") -> None:
        """Compute SSIM map between 'Original' and 'Reconstruction'."""
        steps = self._pipeline_panel.get_all_steps()
        original = None
        reconstruction = None
        for s in steps:
            if "\u539f\u5716" in s.name:
                original = s.array
            if "\u91cd\u5efa" in s.name:
                reconstruction = s.array

        if original is None or reconstruction is None:
            messagebox.showinfo("\u8cc7\u8a0a", "\u9700\u8981\u539f\u5716\u548c\u91cd\u5efa\u5716")
            return

        orig_copy = original.copy()
        recon_copy = reconstruction.copy()
        dev = self.config.device

        def _compute():
            from dl_anomaly.core.anomaly_scorer import AnomalyScorer
            from dl_anomaly.visualization.heatmap import create_error_heatmap

            scorer = AnomalyScorer(device=dev)
            ssim_map = scorer.compute_ssim_map(orig_copy, recon_copy)
            return create_error_heatmap(ssim_map)

        def _done(heatmap):
            self._add_pipeline_step("SSIM \u5716", heatmap)
            self.set_status("SSIM \u5716\u8a08\u7b97\u5b8c\u6210")

        self._run_in_bg(_compute, on_done=_done, status_msg="\u8a08\u7b97 SSIM \u5716...")

    def _cmd_apply_threshold_mask(self: "InspectorApp") -> None:
        """Threshold the error/smoothed map to produce a binary mask."""
        step = self._pipeline_panel.get_current_step()
        if step is None:
            return

        arr = step.array
        if arr.ndim == 3:
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        else:
            gray = arr

        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        self._add_pipeline_step("\u95be\u503c\u906e\u7f69", mask_rgb)

    def _rerun_inspection_with_params(self: "InspectorApp") -> None:
        """Re-run inspection on the current image with updated parameters."""
        if self._inference_pipeline is None or self._current_image_path is None:
            return
        path = self._current_image_path
        if not Path(path).exists():
            return

        self.config.anomaly_threshold_percentile = self._ops_panel.get_threshold_percentile()
        self.config.ssim_weight = self._ops_panel.get_ssim_weight()

        preprocessor = self._inference_pipeline.preprocessor
        model = self._inference_pipeline.model
        device = self._inference_pipeline.device
        dev_str = self.config.device
        ssim_w = self._ops_panel.get_ssim_weight()
        sigma = self._ops_panel.get_sigma()
        min_area = self._ops_panel.get_min_area()
        threshold = self._inference_pipeline.scorer.threshold
        rerun_path = path

        def _compute():
            import torch
            from dl_anomaly.core.anomaly_scorer import AnomalyScorer

            scorer = AnomalyScorer(device=dev_str)

            tensor = preprocessor.load_and_preprocess(rerun_path)
            batch = tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                recon_batch = model(batch)

            orig_np = preprocessor.inverse_normalize(tensor)
            recon_np = preprocessor.inverse_normalize(recon_batch.squeeze(0))

            error_map = scorer.compute_combined_error(orig_np, recon_np, ssim_w)
            smoothed = scorer.create_anomaly_map(error_map, gaussian_sigma=sigma)
            score = scorer.compute_image_score(error_map)

            is_defective = False
            if threshold is not None:
                is_defective = score > threshold

            map_u8 = (smoothed * 255).astype(np.uint8)
            _, mask = cv2.threshold(map_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            filtered_mask = np.zeros_like(mask)
            regions = []
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area >= min_area:
                    filtered_mask[labels == i] = 255
                    x, y, w, h = stats[i, :4]
                    regions.append({"bbox": (int(x), int(y), int(w), int(h)), "area": int(area)})

            from dl_anomaly.pipeline.inference import InspectionResult
            return InspectionResult(
                original=orig_np,
                reconstruction=recon_np,
                error_map=smoothed,
                defect_mask=filtered_mask,
                anomaly_score=score,
                is_defective=is_defective,
                defect_regions=regions,
            )

        def _done(result):
            self._display_inspection_result(result, rerun_path)
            self.set_status(f"\u91cd\u65b0\u6aa2\u6e2c\u5b8c\u6210: {Path(rerun_path).name}")

        self._run_in_bg(_compute, on_done=_done, status_msg="\u91cd\u65b0\u6aa2\u6e2c\u4e2d...")

    # ==================================================================
    # Help
    # ==================================================================

    def _cmd_shortcuts(self: "InspectorApp") -> None:
        shortcuts = (
            f"{display('O'):16s}\u958b\u555f\u5716\u7247\n"
            f"{display('S'):16s}\u5132\u5b58\u5716\u7247\n"
            f"{display('Z'):16s}\u5fa9\u539f\n"
            f"{display('Y'):16s}\u91cd\u505a\n"
            f"{display('I'):16s}\u50cf\u7d20\u503c\u6aa2\u67e5\u5668\u8996\u7a97\n"
            f"{display_shift('I'):16s}\u50cf\u7d20\u6aa2\u67e5\u5de5\u5177\n"
            f"{display_shift('R'):16s}\u5340\u57df\u9078\u53d6\u5de5\u5177\n"
            "Escape          \u8fd4\u56de\u5e73\u79fb\u6a21\u5f0f\n"
            f"{display('T'):16s}\u95be\u503c\u5206\u5272\n"
            "Space           \u7e2e\u653e\u81f3\u7a97\u53e3\n"
            "+/-             \u653e\u5927/\u7e2e\u5c0f\n"
            "F5              \u6aa2\u6e2c\u5716\u7247\n"
            "F6              \u8a13\u7df4\u6a21\u578b\n"
            "F8              \u8173\u672c\u7de8\u8f2f\u5668\n"
            "F9              \u57f7\u884c\u8173\u672c\n"
            f"{DELETE_LABEL:16s}\u522a\u9664\u6b65\u9a5f\n"
            "\n"
            "\u6ed1\u9f20\u5de6\u9375\u62d6\u66f3    \u5e73\u79fb\u5716\u7247 (\u5e73\u79fb\u6a21\u5f0f)\n"
            "\u6ed1\u9f20\u6efe\u8f2a        \u7e2e\u653e\n"
            "\u96d9\u64ca\u5de6\u9375        \u7e2e\u653e\u81f3\u7a97\u53e3\n"
            "\u53f3\u9375            \u5feb\u901f\u9078\u55ae (\u88c1\u5207/\u5206\u5272/\u95be\u503c...)\n"
            "\u4e2d\u9375\u62d6\u66f3        \u7e2e\u653e\u81f3\u9078\u5340"
        )
        messagebox.showinfo("\u5feb\u6377\u9375", shortcuts)

    def _cmd_about(self: "InspectorApp") -> None:
        messagebox.showinfo(
            "\u95dc\u65bc",
            "CV \u7f3a\u9677\u5075\u6e2c\u5668 v2.0 - Industrial Vision Style\n\n"
            "\u6574\u5408\u5f0f\u5de5\u696d\u6aa2\u6e2c\u7cfb\u7d71\n\n"
            "\u2501 DL \u6a21\u5f0f (Autoencoder)\n"
            "  PyTorch CNN \u81ea\u52d5\u7de8\u78bc\u5668 + PatchCore\n"
            "  MSE + SSIM \u7d44\u5408\u640d\u5931\n\n"
            "\u2501 Variation Model \u6a21\u5f0f\n"
            "  Welford \u7dda\u4e0a\u7d71\u8a08\u6f14\u7b97\u6cd5\n"
            "  \u5747\u503c/\u6a19\u6e96\u5dee \u95be\u503c\u5075\u6e2c\n\n"
            "Framework: tkinter + OpenCV + PyTorch",
        )

    # ==================================================================
    # Recipe commands
    # ==================================================================

    def _cmd_save_recipe(self: "InspectorApp") -> None:
        """Export the current pipeline as a recipe JSON file."""
        from dl_anomaly.core.recipe import Recipe

        recipe = Recipe.from_pipeline(self._pipeline_panel)
        if not recipe.steps:
            messagebox.showinfo("\u8cc7\u8a0a", "\u76ee\u524d\u6c92\u6709\u53ef\u5132\u5b58\u7684\u64cd\u4f5c\u6b65\u9a5f\u3002")
            return
        path = filedialog.asksaveasfilename(
            title="\u5132\u5b58\u6d41\u7a0b",
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("\u6240\u6709\u6a94\u6848", "*")],
        )
        if path:
            recipe.save(path)
            self.set_status(f"\u6d41\u7a0b\u5df2\u5132\u5b58: {Path(path).name} ({len(recipe.steps)} \u6b65)")

    def _cmd_load_and_apply_recipe(self: "InspectorApp") -> None:
        """Open the recipe-apply dialog for windowed multi-image recipe application."""
        from dl_anomaly.gui.recipe_apply_dialog import RecipeApplyDialog

        def _add_step(name, arr, region=None):
            self._pipeline_panel.add_step(name, arr, select=False, region=region)
            total = self._pipeline_panel.get_step_count()
            self._pipeline_panel.select_step(total - 1)

        RecipeApplyDialog(self, add_step_cb=_add_step,
                          set_status_cb=self.set_status)

    def _cmd_batch_apply_recipe(self: "InspectorApp") -> None:
        """Batch-apply a recipe to a folder of images."""
        path = filedialog.askopenfilename(
            title="\u9078\u64c7\u6d41\u7a0b\u6a94\u6848",
            filetypes=[("JSON", "*.json")],
        )
        if not path:
            return

        from dl_anomaly.core.recipe import Recipe, replay_recipe

        try:
            recipe = Recipe.load(path)
        except Exception as exc:
            self._show_error("\u7121\u6cd5\u8f09\u5165\u6d41\u7a0b", exc)
            return

        input_dir = filedialog.askdirectory(title="\u9078\u64c7\u8f38\u5165\u5716\u7247\u8cc7\u6599\u593e")
        if not input_dir:
            return

        output_dir = filedialog.askdirectory(title="\u9078\u64c7\u8f38\u51fa\u8cc7\u6599\u593e")
        if not output_dir:
            return

        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        image_paths = sorted(
            str(p) for p in Path(input_dir).rglob("*")
            if p.is_file() and p.suffix.lower() in exts
        )
        if not image_paths:
            messagebox.showinfo("\u8cc7\u8a0a", "\u8cc7\u6599\u593e\u4e2d\u6c92\u6709\u5716\u7247\u6a94\u6848\u3002")
            return

        batch_recipe = recipe
        batch_out_dir = output_dir
        batch_paths = image_paths

        def _compute():
            from PIL import Image as PILImage
            MAX_SIDE = 1080
            processed = 0
            for img_path in batch_paths:
                try:
                    pil = PILImage.open(img_path)
                    w, h = pil.size
                    if max(w, h) > MAX_SIDE:
                        scale = MAX_SIDE / max(w, h)
                        pil = pil.resize(
                            (int(w * scale), int(h * scale)), PILImage.LANCZOS)
                    if pil.mode == "L":
                        arr = np.array(pil)
                    else:
                        arr = np.array(pil.convert("RGB"))

                    results = replay_recipe(batch_recipe, arr)
                    if results:
                        final_name, final_arr, _ = results[-1]
                        if final_arr.ndim == 2:
                            out_img = PILImage.fromarray(final_arr, mode="L")
                        else:
                            out_img = PILImage.fromarray(final_arr, mode="RGB")
                        stem = Path(img_path).stem
                        out_path = Path(batch_out_dir) / f"{stem}_result.png"
                        out_img.save(str(out_path))
                        processed += 1
                except Exception:
                    logger.exception("Batch: failed on %s", img_path)
            return processed

        def _done(count):
            self.set_status(f"\u6279\u6b21\u5957\u7528\u5b8c\u6210: {count}/{len(batch_paths)} \u5f35")
            messagebox.showinfo(
                "\u5b8c\u6210",
                f"\u6279\u6b21\u5957\u7528\u6d41\u7a0b\u5b8c\u6210\u3002\n"
                f"\u8655\u7406: {count}/{len(batch_paths)} \u5f35\n"
                f"\u8f38\u51fa: {batch_out_dir}",
            )

        self._run_in_bg(
            _compute, on_done=_done,
            status_msg=f"\u6279\u6b21\u5957\u7528\u6d41\u7a0b ({len(batch_paths)} \u5f35)...")

    # ==================================================================
    # Settings
    # ==================================================================

    def _cmd_settings(self: "InspectorApp") -> None:
        from dl_anomaly.gui.dialogs import SettingsDialog

        SettingsDialog(self, self.config)
