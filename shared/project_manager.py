"""Project management - save/load complete inspection configurations.

A project is a ZIP archive (.cvproj) containing:
- project.json       -- metadata + all settings
- dl_model.pt        -- DL checkpoint (if exists, copied in)
- vm_model.npz       -- VM model (if exists, copied in)
- recipe.json        -- processing pipeline recipe (if exists)
- rois.json          -- ROI definitions (if exists)
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

PROJECT_EXT = ".cvproj"
PROJECT_VERSION = "1.0"

# Default base directory for extracted projects
_APP_DIR = Path.home() / ".detect_app"
_PROJECTS_DIR = _APP_DIR / "projects"
_RECENT_FILE = _APP_DIR / "recent_projects.json"
_MAX_RECENT = 10

# Internal archive member names
_META_FILE = "project.json"
_DL_MODEL_FILE = "dl_model.pt"
_VM_MODEL_FILE = "vm_model.npz"
_RECIPE_FILE = "recipe.json"
_ROIS_FILE = "rois.json"


@dataclass
class ProjectInfo:
    """Project metadata and settings."""

    name: str = "Untitled"
    description: str = ""
    product_line: str = ""
    created: str = field(default_factory=lambda: datetime.now().isoformat())
    modified: str = field(default_factory=lambda: datetime.now().isoformat())
    version: str = PROJECT_VERSION

    # Detection settings
    dl_config: Dict[str, Any] = field(default_factory=dict)
    vm_config: Dict[str, Any] = field(default_factory=dict)

    # File flags (which models are bundled)
    has_dl_model: bool = False
    has_vm_model: bool = False
    has_recipe: bool = False
    has_rois: bool = False


class ProjectError(Exception):
    """Raised when project save/load operations fail."""


class ProjectManager:
    """Save and load complete inspection project configurations."""

    @staticmethod
    def save_project(
        path: str | Path,
        info: ProjectInfo,
        dl_model_path: Optional[str | Path] = None,
        vm_model_path: Optional[str | Path] = None,
        recipe_data: Optional[Dict] = None,
        roi_data: Optional[List[Dict]] = None,
    ) -> Path:
        """Save a project archive.

        Args:
            path: Output .cvproj file path.
            info: Project metadata.
            dl_model_path: Path to DL checkpoint (.pt) to bundle.
            vm_model_path: Path to VM model (.npz) to bundle.
            recipe_data: Recipe dict to include.
            roi_data: ROI definitions to include.

        Returns:
            Path to saved project file.

        Raises:
            ProjectError: If the save operation fails.
        """
        path = Path(path)
        if path.suffix.lower() != PROJECT_EXT:
            path = path.with_suffix(PROJECT_EXT)

        # Update flags based on provided data
        info.has_dl_model = dl_model_path is not None and Path(dl_model_path).is_file()
        info.has_vm_model = vm_model_path is not None and Path(vm_model_path).is_file()
        info.has_recipe = recipe_data is not None
        info.has_rois = roi_data is not None

        # Update modification timestamp
        info.modified = datetime.now().isoformat()

        tmp_dir = None
        try:
            # Step 1: Create a temporary staging directory
            tmp_dir = Path(tempfile.mkdtemp(prefix="cvproj_"))
            logger.debug("Staging project in %s", tmp_dir)

            # Step 2: Write project.json
            meta_path = tmp_dir / _META_FILE
            meta_path.write_text(
                json.dumps(asdict(info), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            # Step 3: Copy model files if provided
            if info.has_dl_model and dl_model_path is not None:
                src = Path(dl_model_path)
                dst = tmp_dir / _DL_MODEL_FILE
                shutil.copy2(src, dst)
                logger.info("Bundled DL model: %s (%d bytes)", src.name, dst.stat().st_size)

            if info.has_vm_model and vm_model_path is not None:
                src = Path(vm_model_path)
                dst = tmp_dir / _VM_MODEL_FILE
                shutil.copy2(src, dst)
                logger.info("Bundled VM model: %s (%d bytes)", src.name, dst.stat().st_size)

            # Step 4: Write recipe.json and rois.json if provided
            if recipe_data is not None:
                recipe_path = tmp_dir / _RECIPE_FILE
                recipe_path.write_text(
                    json.dumps(recipe_data, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                logger.info("Bundled recipe with %d steps", len(recipe_data.get("steps", [])))

            if roi_data is not None:
                rois_path = tmp_dir / _ROIS_FILE
                rois_path.write_text(
                    json.dumps(roi_data, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                logger.info("Bundled %d ROI definitions", len(roi_data))

            # Step 5: Create ZIP archive
            path.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
                for file_path in sorted(tmp_dir.rglob("*")):
                    if file_path.is_file():
                        arcname = file_path.relative_to(tmp_dir).as_posix()
                        zf.write(file_path, arcname)

            logger.info(
                "Saved project '%s' to %s (%d bytes)",
                info.name,
                path,
                path.stat().st_size,
            )

            # Record in recent projects
            ProjectManager._add_to_recent(path, info.name)

            return path

        except Exception as exc:
            raise ProjectError(f"Failed to save project: {exc}") from exc

        finally:
            # Step 6: Clean up temp directory
            if tmp_dir is not None and tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)

    @staticmethod
    def load_project(
        path: str | Path,
        extract_dir: Optional[str | Path] = None,
    ) -> Tuple[ProjectInfo, Dict[str, Optional[Path]]]:
        """Load a project archive.

        Args:
            path: .cvproj file to load.
            extract_dir: Directory to extract models to.
                         Defaults to ``~/.detect_app/projects/{project_name}/``.

        Returns:
            A tuple of ``(ProjectInfo, paths_dict)`` where ``paths_dict``
            has keys ``'dl_model'``, ``'vm_model'``, ``'recipe'``, ``'rois'``
            -- each ``Optional[Path]`` pointing to the extracted file.

        Raises:
            ProjectError: If the archive is invalid or extraction fails.
        """
        path = Path(path)

        if not path.is_file():
            raise ProjectError(f"Project file not found: {path}")

        # Step 1: Validate the ZIP contains project.json
        try:
            with zipfile.ZipFile(path, "r") as zf:
                names = zf.namelist()
                if _META_FILE not in names:
                    raise ProjectError(
                        f"Invalid project archive: missing {_META_FILE}"
                    )
        except zipfile.BadZipFile as exc:
            raise ProjectError(f"Corrupt project archive: {exc}") from exc

        # Read metadata first to determine project name for extract_dir
        try:
            with zipfile.ZipFile(path, "r") as zf:
                raw_meta = zf.read(_META_FILE).decode("utf-8")
                meta_dict = json.loads(raw_meta)
        except (json.JSONDecodeError, KeyError, UnicodeDecodeError) as exc:
            raise ProjectError(f"Failed to parse project metadata: {exc}") from exc

        # Build ProjectInfo from the loaded dictionary
        info = ProjectInfo()
        for key, value in meta_dict.items():
            if hasattr(info, key):
                setattr(info, key, value)

        # Step 2: Determine extraction directory
        if extract_dir is not None:
            dest = Path(extract_dir)
        else:
            safe_name = "".join(
                c if c.isalnum() or c in ("_", "-") else "_"
                for c in info.name
            ).strip("_") or "project"
            dest = _PROJECTS_DIR / safe_name

        dest.mkdir(parents=True, exist_ok=True)

        # Step 3: Extract all files
        try:
            with zipfile.ZipFile(path, "r") as zf:
                zf.extractall(dest)
        except Exception as exc:
            raise ProjectError(f"Failed to extract project: {exc}") from exc

        # Step 4: Build paths dict for extracted model files
        paths: Dict[str, Optional[Path]] = {
            "dl_model": None,
            "vm_model": None,
            "recipe": None,
            "rois": None,
        }

        dl_path = dest / _DL_MODEL_FILE
        if dl_path.is_file():
            paths["dl_model"] = dl_path

        vm_path = dest / _VM_MODEL_FILE
        if vm_path.is_file():
            paths["vm_model"] = vm_path

        recipe_path = dest / _RECIPE_FILE
        if recipe_path.is_file():
            paths["recipe"] = recipe_path

        rois_path = dest / _ROIS_FILE
        if rois_path.is_file():
            paths["rois"] = rois_path

        logger.info(
            "Loaded project '%s' (v%s) from %s -> %s",
            info.name,
            info.version,
            path,
            dest,
        )

        # Record in recent projects
        ProjectManager._add_to_recent(path, info.name)

        return info, paths

    @staticmethod
    def load_recipe(path: Path) -> Dict:
        """Load a recipe JSON file extracted from a project.

        Args:
            path: Path to recipe.json.

        Returns:
            Recipe dictionary.

        Raises:
            ProjectError: If the file cannot be read or parsed.
        """
        try:
            raw = path.read_text(encoding="utf-8")
            return json.loads(raw)
        except Exception as exc:
            raise ProjectError(f"Failed to load recipe: {exc}") from exc

    @staticmethod
    def load_rois(path: Path) -> List[Dict]:
        """Load ROI definitions from a JSON file extracted from a project.

        Args:
            path: Path to rois.json.

        Returns:
            List of ROI dictionaries.

        Raises:
            ProjectError: If the file cannot be read or parsed.
        """
        try:
            raw = path.read_text(encoding="utf-8")
            return json.loads(raw)
        except Exception as exc:
            raise ProjectError(f"Failed to load ROIs: {exc}") from exc

    # ------------------------------------------------------------------ #
    #  Recent projects                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def list_recent_projects() -> List[Dict[str, str]]:
        """List recently opened projects from app state.

        Returns:
            List of dicts with keys ``'name'``, ``'path'``, ``'timestamp'``.
            Ordered most-recent first.
        """
        if not _RECENT_FILE.is_file():
            return []
        try:
            raw = _RECENT_FILE.read_text(encoding="utf-8")
            entries: List[Dict[str, str]] = json.loads(raw)
            # Filter out entries whose files no longer exist
            valid = [e for e in entries if Path(e.get("path", "")).is_file()]
            return valid
        except Exception:
            logger.warning("Failed to read recent projects from %s", _RECENT_FILE)
            return []

    @staticmethod
    def _add_to_recent(path: Path, name: str) -> None:
        """Add a project file to the recent-projects list."""
        _APP_DIR.mkdir(parents=True, exist_ok=True)

        entries = ProjectManager.list_recent_projects()

        path_str = str(path.resolve())
        # Remove existing entry for same path
        entries = [e for e in entries if e.get("path") != path_str]

        # Prepend new entry
        entries.insert(0, {
            "name": name,
            "path": path_str,
            "timestamp": datetime.now().isoformat(),
        })

        # Trim to max
        entries = entries[:_MAX_RECENT]

        try:
            _RECENT_FILE.write_text(
                json.dumps(entries, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            logger.warning("Failed to write recent projects to %s", _RECENT_FILE)

    @staticmethod
    def clear_recent_projects() -> None:
        """Clear the recent-projects list."""
        if _RECENT_FILE.is_file():
            try:
                _RECENT_FILE.unlink()
            except OSError:
                logger.warning("Failed to clear recent projects file")

    # ------------------------------------------------------------------ #
    #  Validation helpers                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def peek_project(path: str | Path) -> Optional[ProjectInfo]:
        """Read project metadata without fully extracting the archive.

        Args:
            path: .cvproj file to peek into.

        Returns:
            ProjectInfo if valid, None otherwise.
        """
        path = Path(path)
        if not path.is_file():
            return None

        try:
            with zipfile.ZipFile(path, "r") as zf:
                if _META_FILE not in zf.namelist():
                    return None
                raw = zf.read(_META_FILE).decode("utf-8")
                meta_dict = json.loads(raw)

            info = ProjectInfo()
            for key, value in meta_dict.items():
                if hasattr(info, key):
                    setattr(info, key, value)
            return info

        except Exception:
            logger.debug("Failed to peek project at %s", path, exc_info=True)
            return None

    @staticmethod
    def is_valid_project(path: str | Path) -> bool:
        """Check whether a file is a valid .cvproj archive.

        Args:
            path: File path to check.

        Returns:
            True if the file is a valid project archive.
        """
        return ProjectManager.peek_project(path) is not None
