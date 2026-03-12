"""Persistent application state: window geometry, sash positions, recent files."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MAX_RECENT_FILES = 10


class AppState:
    """JSON-backed persistence for window geometry and recent files.

    State is stored at ``~/.detect_app/{app_name}_state.json``.
    """

    def __init__(self, app_name: str) -> None:
        self._app_name = app_name
        self._dir = Path.home() / ".detect_app"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / f"{app_name}_state.json"
        self._data: Dict[str, Any] = self._load()

    # -- I/O ----------------------------------------------------------------

    def _load(self) -> Dict[str, Any]:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text(encoding="utf-8"))
            except Exception:
                logger.warning("Failed to load app state from %s", self._path)
        return {}

    def _save(self) -> None:
        try:
            self._path.write_text(
                json.dumps(self._data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            logger.warning("Failed to save app state to %s", self._path)

    # -- Window geometry ----------------------------------------------------

    def save_geometry(self, root) -> None:
        """Save window geometry string from a Tk root."""
        self._data["geometry"] = root.geometry()
        self._save()

    def restore_geometry(self, root) -> None:
        """Restore previously saved geometry to a Tk root."""
        geo = self._data.get("geometry")
        if geo:
            try:
                root.geometry(geo)
            except Exception:
                pass

    # -- Sash positions -----------------------------------------------------

    def save_sash_positions(self, paned) -> None:
        """Save sash positions of a ttk.PanedWindow."""
        try:
            positions = []
            for i in range(len(paned.panes()) - 1):
                positions.append(paned.sashpos(i))
            self._data["sash_positions"] = positions
            self._save()
        except Exception:
            pass

    def restore_sash_positions(self, paned) -> None:
        """Restore sash positions to a ttk.PanedWindow."""
        positions = self._data.get("sash_positions")
        if positions:
            try:
                for i, pos in enumerate(positions):
                    paned.sashpos(i, pos)
            except Exception:
                pass

    # -- Recent files -------------------------------------------------------

    def get_recent_files(self) -> List[str]:
        """Return list of recent file paths."""
        return list(self._data.get("recent_files", []))

    def add_recent_file(self, path: str) -> None:
        """Add a file to the recent-files list and persist."""
        path = str(path)
        files = self._data.get("recent_files", [])
        if path in files:
            files.remove(path)
        files.insert(0, path)
        self._data["recent_files"] = files[:MAX_RECENT_FILES]
        self._save()
