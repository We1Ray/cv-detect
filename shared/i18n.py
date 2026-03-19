"""Lightweight internationalization (i18n) framework.

Supports Traditional Chinese (zh-TW, default), English (en),
and Simplified Chinese (zh-CN). Uses JSON-based translation files.

Usage::

    from shared.i18n import t, set_language
    set_language("en")
    label = t("inspection.pass")  # "Pass"
    label = t("inspection.pass")  # with zh-TW: "通過"
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_LANGUAGE = "zh-TW"


class Translation:
    """Manages translation loading and key lookup.

    Parameters
    ----------
    locale_dir:
        Directory containing ``<lang>.json`` files.  Defaults to
        ``shared/locales/`` relative to this module.
    """

    def __init__(self, locale_dir: Optional[str] = None) -> None:
        if locale_dir is not None:
            self._locale_dir = Path(locale_dir)
        else:
            self._locale_dir = Path(__file__).parent / "locales"

        self._language: str = _DEFAULT_LANGUAGE
        self._translations: Dict[str, Any] = {}
        self._load_language(self._language)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def set_language(self, lang: str) -> None:
        """Switch the active language.

        Parameters
        ----------
        lang:
            Language code, e.g. ``"zh-TW"``, ``"en"``, ``"zh-CN"``.

        Raises
        ------
        FileNotFoundError
            If the corresponding JSON file does not exist.
        """
        self._load_language(lang)
        self._language = lang

    def get_language(self) -> str:
        """Return the currently active language code."""
        return self._language

    def t(self, key: str, **kwargs: Any) -> str:
        """Translate *key* using dotted notation.

        If the key contains ``{name}``-style placeholders and matching
        *kwargs* are provided, they will be substituted via
        :meth:`str.format`.

        Parameters
        ----------
        key:
            Dotted path, e.g. ``"inspection.pass"``.
        **kwargs:
            Optional format parameters.

        Returns
        -------
        str
            The translated string, or *key* itself if not found.
        """
        parts = key.split(".")
        node: Any = self._translations

        for part in parts:
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                logger.debug("Translation key not found: %s (lang=%s)", key, self._language)
                return key

        if not isinstance(node, str):
            logger.debug("Translation key resolves to non-string: %s", key)
            return key

        if kwargs:
            try:
                return node.format(**kwargs)
            except (KeyError, IndexError):
                logger.debug("Format failed for key %s with kwargs %s", key, kwargs)
                return node

        return node

    def available_languages(self) -> List[str]:
        """Return a sorted list of available language codes.

        Determined by scanning the locale directory for ``*.json`` files.
        """
        if not self._locale_dir.is_dir():
            return []

        langs = sorted(p.stem for p in self._locale_dir.glob("*.json"))
        return langs

    # ------------------------------------------------------------------ #
    #  Internal                                                            #
    # ------------------------------------------------------------------ #

    def _load_language(self, lang: str) -> None:
        """Load translations from the JSON file for *lang*."""
        path = self._locale_dir / f"{lang}.json"
        if not path.exists():
            raise FileNotFoundError(f"Translation file not found: {path}")

        with open(path, encoding="utf-8") as fh:
            self._translations = json.load(fh)

        logger.debug("Loaded translations for %s from %s", lang, path)


# ======================================================================= #
#  Module-level convenience singleton                                      #
# ======================================================================= #

_instance: Optional[Translation] = None


def _get_instance() -> Translation:
    """Lazily create and return the module-level Translation singleton."""
    global _instance
    if _instance is None:
        _instance = Translation()
    return _instance


def t(key: str, **kwargs: Any) -> str:
    """Translate *key* using the global Translation instance.

    See :meth:`Translation.t` for details.
    """
    return _get_instance().t(key, **kwargs)


def set_language(lang: str) -> None:
    """Set the language on the global Translation instance.

    See :meth:`Translation.set_language` for details.
    """
    _get_instance().set_language(lang)


def get_language() -> str:
    """Return the current language from the global Translation instance."""
    return _get_instance().get_language()


def available_languages() -> List[str]:
    """Return available languages from the global Translation instance."""
    return _get_instance().available_languages()
