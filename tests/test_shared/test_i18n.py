"""Tests for the shared.i18n internationalization module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from shared.i18n import Translation


@pytest.fixture()
def translator() -> Translation:
    """Return a Translation instance using the default locale directory."""
    return Translation()


@pytest.fixture()
def custom_translator(tmp_path: Path) -> Translation:
    """Return a Translation instance with a custom locale directory
    containing a placeholder-enabled translation.
    """
    data = {
        "greeting": "Hello, {name}!",
        "nested": {"welcome": "Welcome to {place}, {name}."},
    }
    # Write a minimal zh-TW file so the constructor can load its default.
    default_file = tmp_path / "zh-TW.json"
    default_file.write_text(json.dumps({}), encoding="utf-8")

    locale_file = tmp_path / "en.json"
    locale_file.write_text(json.dumps(data), encoding="utf-8")
    tr = Translation(locale_dir=str(tmp_path))
    tr.set_language("en")
    return tr


class TestTranslation:
    """Unit tests for the Translation class."""

    def test_default_language(self, translator: Translation) -> None:
        """Default language is zh-TW."""
        assert translator.get_language() == "zh-TW"

    def test_translate_known_key(self, translator: Translation) -> None:
        """Translating a known key returns the correct zh-TW string."""
        assert translator.t("inspection.pass") == "通過"

    def test_translate_nested_key(self, translator: Translation) -> None:
        """Dotted keys resolve through nested dictionaries."""
        assert translator.t("menu.file.open") == "開啟圖片"

    def test_translate_unknown_key(self, translator: Translation) -> None:
        """An unknown key is returned unchanged as a fallback."""
        key = "this.key.does.not.exist"
        assert translator.t(key) == key

    def test_set_language_en(self, translator: Translation) -> None:
        """Switching to English returns English translations."""
        translator.set_language("en")
        assert translator.get_language() == "en"
        assert translator.t("inspection.pass") == "Pass"
        assert translator.t("app.title") == "DL Anomaly Detector"

    def test_set_language_zh_cn(self, translator: Translation) -> None:
        """Switching to Simplified Chinese returns zh-CN translations."""
        translator.set_language("zh-CN")
        assert translator.get_language() == "zh-CN"
        assert translator.t("inspection.pass") == "通过"
        assert translator.t("menu.file.open") == "打开图片"

    def test_set_language_invalid(self, translator: Translation) -> None:
        """Setting an unsupported language raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            translator.set_language("fr-FR")

    def test_format_params(self, custom_translator: Translation) -> None:
        """Placeholders in translation strings are substituted."""
        result = custom_translator.t("greeting", name="World")
        assert result == "Hello, World!"

        result = custom_translator.t("nested.welcome", name="Alice", place="Taipei")
        assert result == "Welcome to Taipei, Alice."

    def test_available_languages(self, translator: Translation) -> None:
        """available_languages returns at least the three shipped locales."""
        langs = translator.available_languages()
        assert isinstance(langs, list)
        assert len(langs) >= 3
        assert "en" in langs
        assert "zh-TW" in langs
        assert "zh-CN" in langs

    def test_key_resolves_to_dict_returns_key(self, translator: Translation) -> None:
        """If a key resolves to a dict (not a leaf), return the key itself."""
        # "menu.file" is a dict with open/save/exit children.
        result = translator.t("menu.file")
        assert result == "menu.file"
