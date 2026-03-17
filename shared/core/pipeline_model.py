"""Pipeline Model - treat an entire inspection pipeline as a single model artifact.

A ``PipelineModel`` bundles preprocessing recipe, inspection flow, model
checkpoints, template images, and calibration files into one portable
``.cpmodel`` ZIP archive.  It can be saved, loaded, shared, and executed
as a single unit — just like loading a trained model.

Typical usage::

    # Build and save
    model = PipelineModel.build(
        name="PCB Inspector v2",
        recipe=my_recipe,
        flow=my_flow,
        author="QA Team",
        description="PCB solder joint inspection pipeline",
    )
    model.save("pcb_inspector_v2.cpmodel")

    # Load and execute
    with PipelineModel.load("pcb_inspector_v2.cpmodel") as model:
        result = model.execute(image)
        print(result.overall_pass)

File format (``.cpmodel`` is a ZIP archive)::

    manifest.json       # metadata + pipeline config + format version
    recipe.json         # optional preprocessing recipe
    flow.json           # inspection flow definition (paths relativised)
    models/             # .pt, .onnx, .npz checkpoint files
    templates/          # template images for shape matching
    calibration/        # calibration JSON files
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import threading
import time
import zipfile
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from shared.op_logger import log_operation

logger = logging.getLogger(__name__)

# ====================================================================== #
#  Constants                                                               #
# ====================================================================== #

FORMAT_VERSION = "1.0"

MANIFEST_FILE = "manifest.json"
RECIPE_FILE = "recipe.json"
FLOW_FILE = "flow.json"

# Config keys in InspectionFlow steps that reference external files,
# mapped to the subdirectory they should be stored under inside the
# archive.
_PATH_KEY_TO_ARCHIVE_DIR: Dict[str, str] = {
    "checkpoint_path": "models",
    "model_path": "models",
    "template_path": "templates",
    "calibration_path": "calibration",
}

# Binary extensions that are already compressed — store them without
# additional ZIP compression to avoid wasting CPU.
_STORE_EXTENSIONS = frozenset({".pt", ".onnx", ".npz", ".pth", ".bin"})

# File extensions that look like paths (heuristic for validate warnings).
_FILE_EXTENSIONS = frozenset({
    ".pt", ".onnx", ".npz", ".pth", ".bin", ".png", ".jpg", ".jpeg",
    ".bmp", ".tif", ".tiff", ".json", ".yaml", ".yml",
})


# ====================================================================== #
#  Path manipulation helpers                                               #
# ====================================================================== #


def _collect_embedded_files(
    flow_dict: Dict[str, Any],
) -> List[Tuple[int, str, str]]:
    """Walk step configs and find path-valued keys to embed.

    Returns
    -------
    List of ``(step_index, config_key, absolute_path)`` tuples.
    """
    results: List[Tuple[int, str, str]] = []
    for idx, step in enumerate(flow_dict.get("steps", [])):
        config = step.get("config", {})
        for key, archive_dir in _PATH_KEY_TO_ARCHIVE_DIR.items():
            value = config.get(key, "")
            if value and isinstance(value, str) and os.path.isfile(value):
                results.append((idx, key, value))
    return results


def _relativize_paths(
    flow_dict: Dict[str, Any],
    file_mapping: Dict[str, str],
) -> Dict[str, Any]:
    """Replace absolute paths in step configs with archive-relative paths.

    Parameters
    ----------
    flow_dict:
        Serialised flow dict (will NOT be mutated — a deep copy is made).
    file_mapping:
        ``{original_absolute_path: archive_relative_path}``
    """
    result = deepcopy(flow_dict)
    for step in result.get("steps", []):
        config = step.get("config", {})
        for key in _PATH_KEY_TO_ARCHIVE_DIR:
            value = config.get(key, "")
            if value and value in file_mapping:
                config[key] = file_mapping[value]
    return result


def _absolutize_paths(
    flow_dict: Dict[str, Any],
    extract_dir: str,
) -> Dict[str, Any]:
    """Rewrite archive-relative paths to absolute paths under *extract_dir*.

    Parameters
    ----------
    flow_dict:
        Serialised flow dict (will NOT be mutated — a deep copy is made).
    extract_dir:
        Root directory where the archive was extracted.
    """
    result = deepcopy(flow_dict)
    archive_dirs = set(_PATH_KEY_TO_ARCHIVE_DIR.values())
    for step in result.get("steps", []):
        config = step.get("config", {})
        for key in _PATH_KEY_TO_ARCHIVE_DIR:
            value = config.get(key, "")
            if not value or not isinstance(value, str):
                continue
            # Check if the value starts with one of the known archive dirs.
            parts = value.replace("\\", "/").split("/", 1)
            if parts[0] in archive_dirs:
                config[key] = os.path.join(extract_dir, value)
    return result


# ====================================================================== #
#  PipelineModel                                                           #
# ====================================================================== #


class PipelineModel:
    """Portable inspection pipeline packaged as a single ``.cpmodel`` file.

    Parameters
    ----------
    metadata:
        Model metadata (name, version, author, description, etc.).
    recipe:
        Optional preprocessing recipe.
    flow:
        The inspection flow to execute.
    pipeline_config:
        Configuration for parallel batch execution.
    """

    def __init__(
        self,
        metadata: Optional[Dict[str, Any]] = None,
        recipe: Any = None,
        flow: Any = None,
        pipeline_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.metadata: Dict[str, Any] = metadata or {}
        self.recipe = recipe  # Optional[Recipe]
        self.flow = flow  # Optional[InspectionFlow]
        self.pipeline_config: Dict[str, Any] = pipeline_config or {
            "num_preprocess_workers": 2,
            "num_postprocess_workers": 2,
            "max_queue_size": 10,
            "stage_timeout": 30.0,
        }
        self._extract_dir: Optional[str] = None
        self._pipeline_lock = threading.Lock()

    # ------------------------------------------------------------------ #
    #  Factory                                                             #
    # ------------------------------------------------------------------ #

    @classmethod
    def build(
        cls,
        name: str,
        recipe: Any = None,
        flow: Any = None,
        author: str = "",
        description: str = "",
        target_product: str = "",
        version: str = "1.0.0",
        pipeline_config: Optional[Dict[str, Any]] = None,
    ) -> "PipelineModel":
        """Create a PipelineModel from components.

        Parameters
        ----------
        name:
            Human-readable model name.
        recipe:
            Optional :class:`Recipe` for preprocessing.
        flow:
            :class:`InspectionFlow` defining the inspection steps.
        author:
            Author name or team.
        description:
            Free-text description.
        target_product:
            Product or part this pipeline is designed for.
        version:
            Semantic version string.
        pipeline_config:
            Parallel pipeline settings.
        """
        metadata = {
            "name": name,
            "version": version,
            "author": author,
            "description": description,
            "target_product": target_product,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "format_version": FORMAT_VERSION,
        }
        return cls(
            metadata=metadata,
            recipe=recipe,
            flow=flow,
            pipeline_config=pipeline_config,
        )

    # ------------------------------------------------------------------ #
    #  Save                                                                #
    # ------------------------------------------------------------------ #

    @log_operation(logger)
    def save(self, path: str) -> None:
        """Save the pipeline model to a ``.cpmodel`` ZIP archive.

        All referenced external files (checkpoints, templates, calibration)
        are embedded into the archive so that the ``.cpmodel`` is fully
        self-contained and portable.

        Parameters
        ----------
        path:
            Destination file path (will be created / overwritten).
        """
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(str(dest), "w") as zf:
            # 1. Manifest
            manifest = {
                "format_version": FORMAT_VERSION,
                "metadata": self.metadata,
                "pipeline_config": self.pipeline_config,
                "has_recipe": self.recipe is not None,
                "has_flow": self.flow is not None,
            }
            zf.writestr(
                MANIFEST_FILE,
                json.dumps(manifest, indent=2, ensure_ascii=False),
                compress_type=zipfile.ZIP_DEFLATED,
            )

            # 2. Recipe (optional)
            if self.recipe is not None:
                recipe_data = {
                    "version": getattr(self.recipe, "version", 1),
                    "steps": getattr(self.recipe, "steps", []),
                }
                zf.writestr(
                    RECIPE_FILE,
                    json.dumps(recipe_data, indent=2, ensure_ascii=False),
                    compress_type=zipfile.ZIP_DEFLATED,
                )

            # 3. Flow + embedded files
            if self.flow is not None:
                flow_dict = self._flow_to_dict(self.flow)
                embedded = _collect_embedded_files(flow_dict)

                # Build mapping: original path -> archive relative path
                file_mapping: Dict[str, str] = {}
                used_names: Dict[str, int] = {}

                for step_idx, config_key, abs_path in embedded:
                    archive_dir = _PATH_KEY_TO_ARCHIVE_DIR[config_key]
                    basename = os.path.basename(abs_path)

                    # Disambiguate duplicate basenames
                    unique_key = f"{archive_dir}/{basename}"
                    if unique_key in used_names:
                        used_names[unique_key] += 1
                        name_stem, ext = os.path.splitext(basename)
                        basename = f"{name_stem}_step{step_idx}{ext}"
                    else:
                        used_names[unique_key] = 1

                    archive_path = f"{archive_dir}/{basename}"
                    file_mapping[abs_path] = archive_path

                    # Write the file into the ZIP
                    ext = os.path.splitext(abs_path)[1].lower()
                    compress = (
                        zipfile.ZIP_STORED
                        if ext in _STORE_EXTENSIONS
                        else zipfile.ZIP_DEFLATED
                    )
                    zf.write(abs_path, archive_path, compress_type=compress)

                # Relativize paths in flow config
                flow_dict = _relativize_paths(flow_dict, file_mapping)
                zf.writestr(
                    FLOW_FILE,
                    json.dumps(flow_dict, indent=2, ensure_ascii=False),
                    compress_type=zipfile.ZIP_DEFLATED,
                )

        logger.info(
            "PipelineModel saved to %s (%d embedded files).",
            path,
            len(embedded) if self.flow else 0,
        )

    # ------------------------------------------------------------------ #
    #  Load                                                                #
    # ------------------------------------------------------------------ #

    @classmethod
    def load(cls, path: str) -> "PipelineModel":
        """Load a pipeline model from a ``.cpmodel`` ZIP archive.

        The archive is extracted to a temporary directory.  Use the model
        as a context manager (``with PipelineModel.load(...) as m:``) to
        ensure cleanup, or call :meth:`close` explicitly.

        Parameters
        ----------
        path:
            Path to the ``.cpmodel`` file.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        zipfile.BadZipFile
            If the file is not a valid ZIP archive.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Pipeline model not found: {path}")

        extract_dir = tempfile.mkdtemp(prefix="cpmodel_")

        with zipfile.ZipFile(str(p), "r") as zf:
            zf.extractall(extract_dir)

        # 1. Read manifest
        manifest_path = os.path.join(extract_dir, MANIFEST_FILE)
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        metadata = manifest.get("metadata", {})
        pipeline_config = manifest.get("pipeline_config", {})

        # 2. Load recipe (optional)
        recipe = None
        recipe_path = os.path.join(extract_dir, RECIPE_FILE)
        if os.path.exists(recipe_path):
            from dl_anomaly.core.recipe import Recipe

            recipe = Recipe.load(recipe_path)

        # 3. Load flow with absolutized paths
        flow = None
        flow_path = os.path.join(extract_dir, FLOW_FILE)
        if os.path.exists(flow_path):
            with open(flow_path, "r", encoding="utf-8") as f:
                flow_dict = json.load(f)

            # Rewrite relative paths to absolute paths under extract_dir
            flow_dict = _absolutize_paths(flow_dict, extract_dir)

            # Write the rewritten flow JSON and load via InspectionFlow
            rewritten_path = os.path.join(extract_dir, "_flow_abs.json")
            with open(rewritten_path, "w", encoding="utf-8") as f:
                json.dump(flow_dict, f, indent=2, ensure_ascii=False)

            from dl_anomaly.core.inspection_flow import InspectionFlow

            flow = InspectionFlow.load(rewritten_path)

        instance = cls(
            metadata=metadata,
            recipe=recipe,
            flow=flow,
            pipeline_config=pipeline_config,
        )
        instance._extract_dir = extract_dir

        logger.info(
            "PipelineModel loaded from %s (extract_dir=%s).",
            path,
            extract_dir,
        )
        return instance

    # ------------------------------------------------------------------ #
    #  Execution                                                           #
    # ------------------------------------------------------------------ #

    @log_operation(logger)
    def execute(self, image: np.ndarray) -> Any:
        """Run the full pipeline on a single image.

        Steps:
        1. Apply preprocessing recipe (if present).
        2. Execute the inspection flow.

        Parameters
        ----------
        image:
            Input image (BGR uint8 or grayscale).

        Returns
        -------
        FlowResult
            The inspection result.

        Raises
        ------
        RuntimeError
            If no inspection flow is configured.
        """
        if self.flow is None:
            raise RuntimeError("No inspection flow configured in this model.")

        preprocessed = self._apply_recipe(image)
        return self.flow.execute(preprocessed)

    def execute_batch(
        self,
        images: List[Tuple[str, np.ndarray]],
        parallel: bool = False,
        on_result: Optional[Callable] = None,
    ) -> List[Any]:
        """Execute the pipeline on multiple images.

        Parameters
        ----------
        images:
            List of ``(image_id, image_array)`` tuples.
        parallel:
            When ``True``, use a ``ParallelPipeline`` with thread pool
            workers for throughput.  When ``False`` (default), process
            sequentially.
        on_result:
            Optional callback ``(index, FlowResult)`` after each image.

        Returns
        -------
        List[FlowResult]
        """
        if self.flow is None:
            raise RuntimeError("No inspection flow configured in this model.")

        if not parallel:
            return self.flow.execute_batch(images, on_result=on_result)

        # Parallel mode: use create_batch_processor
        from shared.core.parallel_pipeline import create_batch_processor

        num_workers = self.pipeline_config.get("num_preprocess_workers", 2)
        timeout = self.pipeline_config.get("stage_timeout", 30.0)

        def _process_single(item: Tuple[str, np.ndarray]) -> Any:
            img_id, img = item
            preprocessed = self._apply_recipe(img)
            result = self.flow.execute(preprocessed)
            result.source_image_path = img_id
            return result

        pipeline = create_batch_processor(
            process_func=_process_single,
            num_workers=num_workers,
            timeout=timeout,
        )

        pipeline_results = pipeline.process_batch(images)

        results = []
        for idx, pr in enumerate(pipeline_results):
            flow_result = pr.output
            results.append(flow_result)
            if on_result is not None:
                try:
                    on_result(idx, flow_result)
                except Exception:
                    logger.warning(
                        "on_result callback failed for image %d.",
                        idx,
                        exc_info=True,
                    )

        return results

    # ------------------------------------------------------------------ #
    #  Info & validation                                                   #
    # ------------------------------------------------------------------ #

    def info(self) -> Dict[str, Any]:
        """Return a summary of the model metadata and configuration."""
        summary = dict(self.metadata)
        summary["has_recipe"] = self.recipe is not None
        summary["has_flow"] = self.flow is not None
        if self.flow is not None:
            summary["num_steps"] = len(self.flow)
            summary["step_names"] = [
                s.name for s in self.flow.get_steps()
            ]
        summary["pipeline_config"] = dict(self.pipeline_config)
        return summary

    def validate(self) -> List[str]:
        """Check the model configuration for potential issues.

        Returns
        -------
        List[str]
            Warning messages.  Empty list means no issues.
        """
        warnings: List[str] = []

        if self.flow is None:
            warnings.append("No inspection flow configured.")
            return warnings

        # Validate the flow itself.
        flow_warnings = self.flow.validate()
        warnings.extend(flow_warnings)

        # Check that all referenced files exist on disk.
        for step in self.flow.get_steps():
            for key in _PATH_KEY_TO_ARCHIVE_DIR:
                value = step.config.get(key, "")
                if value and isinstance(value, str):
                    if not os.path.isfile(value):
                        warnings.append(
                            f"Step '{step.name}': {key}='{value}' "
                            f"file does not exist."
                        )

        # Check pipeline_config sanity.
        for key in ("num_preprocess_workers", "num_postprocess_workers"):
            val = self.pipeline_config.get(key, 0)
            if not isinstance(val, int) or val < 1:
                warnings.append(
                    f"pipeline_config['{key}']={val} should be >= 1."
                )

        return warnings

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def close(self) -> None:
        """Clean up the temporary extraction directory."""
        if self._extract_dir and os.path.isdir(self._extract_dir):
            shutil.rmtree(self._extract_dir, ignore_errors=True)
            logger.debug("Cleaned up extract dir: %s", self._extract_dir)
            self._extract_dir = None

    def __enter__(self) -> "PipelineModel":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        name = self.metadata.get("name", "unnamed")
        ver = self.metadata.get("version", "?")
        n_steps = len(self.flow) if self.flow else 0
        return f"<PipelineModel name={name!r} v{ver} steps={n_steps}>"

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _apply_recipe(self, image: np.ndarray) -> np.ndarray:
        """Apply preprocessing recipe if present, return processed image."""
        if self.recipe is None:
            return image

        from dl_anomaly.core.recipe import replay_recipe

        results = replay_recipe(self.recipe, image)
        if results:
            # Use the last step's output image.
            _, last_image, _ = results[-1]
            return last_image
        return image

    @staticmethod
    def _flow_to_dict(flow: Any) -> Dict[str, Any]:
        """Serialise an InspectionFlow to a dictionary (without writing
        to disk)."""
        return {
            "flow_name": flow.name,
            "stop_on_failure": flow.stop_on_failure,
            "steps": [s.to_dict() for s in flow.get_steps()],
            "version": "1.0",
        }


# ====================================================================== #
#  PipelineModelRegistry                                                   #
# ====================================================================== #


class PipelineModelRegistry:
    """Manages a directory of ``.cpmodel`` files.

    Provides listing, searching, loading, and deletion without needing
    to fully extract each archive.

    Parameters
    ----------
    directory:
        Root directory to scan for ``.cpmodel`` files.
    """

    def __init__(self, directory: str) -> None:
        self._directory = Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, Dict[str, Any]] = {}

    def scan(self) -> None:
        """Scan the directory for ``.cpmodel`` files and cache manifests."""
        self._cache.clear()
        for f in self._directory.glob("*.cpmodel"):
            try:
                with zipfile.ZipFile(str(f), "r") as zf:
                    with zf.open(MANIFEST_FILE) as mf:
                        manifest = json.loads(mf.read().decode("utf-8"))
                self._cache[f.name] = {
                    "filename": f.name,
                    "path": str(f),
                    "metadata": manifest.get("metadata", {}),
                    "pipeline_config": manifest.get("pipeline_config", {}),
                    "file_size_mb": round(f.stat().st_size / (1024 * 1024), 2),
                }
            except Exception:
                logger.warning(
                    "Failed to read manifest from %s", f, exc_info=True
                )

    def list_models(self) -> List[Dict[str, Any]]:
        """Return cached model manifests, sorted by creation date (newest first)."""
        if not self._cache:
            self.scan()

        entries = list(self._cache.values())
        entries.sort(
            key=lambda e: e.get("metadata", {}).get("created_at", ""),
            reverse=True,
        )
        return entries

    def get_model(self, filename: str) -> PipelineModel:
        """Load a model by filename.

        Parameters
        ----------
        filename:
            The ``.cpmodel`` filename (e.g. ``"pcb_v2.cpmodel"``).

        Raises
        ------
        FileNotFoundError
            If the file does not exist in the registry directory.
        """
        path = self._directory / filename
        return PipelineModel.load(str(path))

    def find_model(self, name: str) -> Optional[Dict[str, Any]]:
        """Find a model entry by metadata name (case-insensitive).

        Returns the cache entry or ``None``.
        """
        if not self._cache:
            self.scan()
        name_lower = name.lower()
        for entry in self._cache.values():
            if entry.get("metadata", {}).get("name", "").lower() == name_lower:
                return entry
        return None

    def add_model(
        self,
        model: PipelineModel,
        filename: Optional[str] = None,
        overwrite: bool = False,
    ) -> str:
        """Save a model to the registry directory.

        Parameters
        ----------
        model:
            The PipelineModel to save.
        filename:
            Target filename.  If ``None``, derived from model name.
        overwrite:
            Whether to overwrite an existing file.

        Returns
        -------
        str
            The full path of the saved file.

        Raises
        ------
        FileExistsError
            If the file already exists and *overwrite* is ``False``.
        """
        if filename is None:
            safe_name = (
                model.metadata.get("name", "model")
                .replace(" ", "_")
                .replace("/", "_")
            )
            filename = f"{safe_name}.cpmodel"

        dest = self._directory / filename
        if dest.exists() and not overwrite:
            raise FileExistsError(f"Model already exists: {dest}")

        model.save(str(dest))

        # Refresh cache entry.
        self._cache[filename] = {
            "filename": filename,
            "path": str(dest),
            "metadata": dict(model.metadata),
            "pipeline_config": dict(model.pipeline_config),
            "file_size_mb": round(dest.stat().st_size / (1024 * 1024), 2),
        }

        return str(dest)

    def delete_model(self, filename: str) -> None:
        """Delete a model from the registry directory.

        Parameters
        ----------
        filename:
            The ``.cpmodel`` filename to delete.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        """
        path = self._directory / filename
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        path.unlink()
        self._cache.pop(filename, None)
        logger.info("Deleted model: %s", path)

    def __repr__(self) -> str:
        return (
            f"<PipelineModelRegistry dir={str(self._directory)!r} "
            f"models={len(self._cache)}>"
        )
