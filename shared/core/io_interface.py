"""Industrial I/O communication interface for PLC integration.

Provides an abstract interface and stub implementations for:
- Digital I/O signals (trigger in, OK/NG out)
- Modbus TCP communication
- OPC-UA client (stub)

This module defines the interface that production line integrators
can implement for their specific hardware.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Set

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class IOInterface(ABC):
    """Abstract interface for industrial I/O communication.

    Subclasses implement the concrete transport (Modbus TCP, OPC-UA,
    digital I/O cards, etc.).  All public methods are designed to be
    called from any thread; implementations should be thread-safe.
    """

    @abstractmethod
    def connect(self) -> None:
        """Establish the connection to the I/O device.

        Raises
        ------
        ConnectionError
            If the connection cannot be established.
        """

    @abstractmethod
    def disconnect(self) -> None:
        """Gracefully close the connection."""

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Return True if the connection is currently active."""

    @abstractmethod
    def wait_for_trigger(self, timeout_ms: int = 5000) -> bool:
        """Block until an inspection trigger signal is received.

        Parameters
        ----------
        timeout_ms:
            Maximum time to wait in milliseconds.  A value of ``0``
            means check once without blocking.

        Returns
        -------
        bool
            True if a trigger was received within the timeout, False
            if the wait timed out.
        """

    @abstractmethod
    def send_result(self, is_pass: bool, score: float) -> None:
        """Send an OK/NG inspection result to the PLC.

        Parameters
        ----------
        is_pass:
            True for OK (good part), False for NG (defect).
        score:
            A continuous quality score (0.0 -- 1.0) that can be written
            to an analog / holding register for SPC trending.
        """

    @abstractmethod
    def read_register(self, address: int) -> int:
        """Read a single 16-bit holding register.

        Parameters
        ----------
        address:
            Register address (0-based).

        Returns
        -------
        int
            The register value (0 -- 65535).
        """

    @abstractmethod
    def write_register(self, address: int, value: int) -> None:
        """Write a single 16-bit holding register.

        Parameters
        ----------
        address:
            Register address (0-based).
        value:
            Value to write (0 -- 65535).
        """


# ---------------------------------------------------------------------------
# StubIOInterface -- for development and testing
# ---------------------------------------------------------------------------

class StubIOInterface(IOInterface):
    """Stub implementation that logs all calls and always succeeds.

    Useful during development and GUI testing when no real PLC hardware
    is available.  The trigger always fires immediately.
    """

    def __init__(self) -> None:
        self._connected: bool = False
        self._registers: dict[int, int] = {}

    def connect(self) -> None:
        self._connected = True
        logger.info("[StubIO] Connected (simulated).")

    def disconnect(self) -> None:
        self._connected = False
        logger.info("[StubIO] Disconnected (simulated).")

    @property
    def is_connected(self) -> bool:
        return self._connected

    def wait_for_trigger(self, timeout_ms: int = 5000) -> bool:
        logger.debug(
            "[StubIO] wait_for_trigger(timeout_ms=%d) -> True (immediate).",
            timeout_ms,
        )
        return True

    def send_result(self, is_pass: bool, score: float) -> None:
        label = "OK" if is_pass else "NG"
        logger.info(
            "[StubIO] send_result: %s  score=%.4f", label, score,
        )

    def read_register(self, address: int) -> int:
        value = self._registers.get(address, 0)
        logger.debug(
            "[StubIO] read_register(%d) -> %d", address, value,
        )
        return value

    def write_register(self, address: int, value: int) -> None:
        self._registers[address] = value
        logger.debug(
            "[StubIO] write_register(%d, %d)", address, value,
        )


# ---------------------------------------------------------------------------
# ModbusTCPInterface
# ---------------------------------------------------------------------------

# Default Modbus register map -- integrators can override via constructor.
_DEFAULT_TRIGGER_COIL: int = 0       # Coil 0: trigger input
_DEFAULT_RESULT_REGISTER: int = 100  # HR 100: 1=OK, 2=NG
_DEFAULT_SCORE_REGISTER: int = 101   # HR 101: score * 10000 (int16)


class ModbusTCPInterface(IOInterface):
    """Modbus TCP client for PLC communication.

    Requires the ``pymodbus`` package.  If it is not installed, the
    class can still be instantiated but :meth:`connect` will raise
    ``ImportError``.

    Parameters
    ----------
    host:
        PLC IP address or hostname.
    port:
        Modbus TCP port (default 502).
    unit_id:
        Modbus unit / slave ID (default 1).
    trigger_coil:
        Coil address to poll for the inspection trigger.
    result_register:
        Holding register to write OK (1) or NG (2).
    score_register:
        Holding register to write the quality score (scaled to int).
    """

    def __init__(
        self,
        host: str,
        port: int = 502,
        unit_id: int = 1,
        trigger_coil: int = _DEFAULT_TRIGGER_COIL,
        result_register: int = _DEFAULT_RESULT_REGISTER,
        score_register: int = _DEFAULT_SCORE_REGISTER,
    ) -> None:
        self._host = host
        self._port = port
        self._unit_id = unit_id
        self._trigger_coil = trigger_coil
        self._result_register = result_register
        self._score_register = score_register

        self._client: Optional[object] = None  # pymodbus client instance
        self._connected: bool = False

    def connect(self) -> None:
        try:
            from pymodbus.client import ModbusTcpClient
        except ImportError as exc:
            raise ImportError(
                "pymodbus is required for ModbusTCPInterface. "
                "Install it with: pip install pymodbus"
            ) from exc

        self._client = ModbusTcpClient(self._host, port=self._port)
        if not self._client.connect():  # type: ignore[union-attr]
            raise ConnectionError(
                f"Cannot connect to Modbus device at "
                f"{self._host}:{self._port}"
            )
        self._connected = True
        logger.info(
            "[ModbusTCP] Connected to %s:%d (unit=%d).",
            self._host, self._port, self._unit_id,
        )

    def disconnect(self) -> None:
        if self._client is not None:
            self._client.close()  # type: ignore[union-attr]
        self._connected = False
        logger.info("[ModbusTCP] Disconnected.")

    @property
    def is_connected(self) -> bool:
        return self._connected

    def wait_for_trigger(self, timeout_ms: int = 5000) -> bool:
        """Poll the trigger coil until it reads True or timeout expires."""
        if self._client is None:
            logger.warning("[ModbusTCP] Not connected; cannot wait for trigger.")
            return False

        deadline = time.monotonic() + timeout_ms / 1000.0
        poll_interval = 0.01  # 10 ms

        while time.monotonic() < deadline:
            try:
                result = self._client.read_coils(  # type: ignore[union-attr]
                    self._trigger_coil, 1, slave=self._unit_id,
                )
                if not result.isError() and result.bits[0]:
                    # Auto-reset the coil so the next cycle requires a new
                    # rising edge from the PLC.
                    self._client.write_coil(  # type: ignore[union-attr]
                        self._trigger_coil, False, slave=self._unit_id,
                    )
                    logger.debug("[ModbusTCP] Trigger received.")
                    return True
            except Exception:
                logger.warning(
                    "[ModbusTCP] Error polling trigger coil.", exc_info=True,
                )
                return False
            time.sleep(poll_interval)

        logger.debug("[ModbusTCP] Trigger timeout (%d ms).", timeout_ms)
        return False

    def send_result(self, is_pass: bool, score: float) -> None:
        if self._client is None:
            logger.warning("[ModbusTCP] Not connected; cannot send result.")
            return

        result_value = 1 if is_pass else 2
        score_int = max(0, min(10000, int(score * 10000)))

        try:
            self._client.write_register(  # type: ignore[union-attr]
                self._result_register, result_value, slave=self._unit_id,
            )
            self._client.write_register(  # type: ignore[union-attr]
                self._score_register, score_int, slave=self._unit_id,
            )
            label = "OK" if is_pass else "NG"
            logger.info(
                "[ModbusTCP] Result sent: %s  score=%d (%.4f).",
                label, score_int, score,
            )
        except Exception:
            logger.error(
                "[ModbusTCP] Failed to send result.", exc_info=True,
            )

    def read_register(self, address: int) -> int:
        if self._client is None:
            logger.warning("[ModbusTCP] Not connected; cannot read register.")
            return 0
        try:
            result = self._client.read_holding_registers(  # type: ignore[union-attr]
                address, 1, slave=self._unit_id,
            )
            if result.isError():
                logger.warning(
                    "[ModbusTCP] Error reading register %d: %s",
                    address, result,
                )
                return 0
            value = result.registers[0]
            logger.debug("[ModbusTCP] read_register(%d) -> %d", address, value)
            return value
        except Exception:
            logger.error(
                "[ModbusTCP] Exception reading register %d.", address,
                exc_info=True,
            )
            return 0

    def write_register(self, address: int, value: int) -> None:
        if self._client is None:
            logger.warning("[ModbusTCP] Not connected; cannot write register.")
            return
        try:
            self._client.write_register(  # type: ignore[union-attr]
                address, value, slave=self._unit_id,
            )
            logger.debug(
                "[ModbusTCP] write_register(%d, %d)", address, value,
            )
        except Exception:
            logger.error(
                "[ModbusTCP] Exception writing register %d.", address,
                exc_info=True,
            )


# ---------------------------------------------------------------------------
# FileWatchInterface -- trigger from filesystem (testing without hardware)
# ---------------------------------------------------------------------------

class FileWatchInterface(IOInterface):
    """File-system watcher that treats new image files as triggers.

    Monitors *watch_dir* for newly created files matching *extensions*.
    Each call to :meth:`wait_for_trigger` returns ``True`` when a new
    file appears and stores its path in :attr:`last_trigger_path` so the
    caller can load and inspect it.

    This is useful for integration testing without a PLC -- simply copy
    or save an image into the watch directory.

    Parameters
    ----------
    watch_dir:
        Directory to watch for new files.
    extensions:
        File suffixes to consider (case-insensitive).
    """

    def __init__(
        self,
        watch_dir: str,
        extensions: Optional[List[str]] = None,
    ) -> None:
        self._watch_dir = Path(watch_dir)
        self._extensions: frozenset[str] = frozenset(
            ext.lower() for ext in (extensions or [".png", ".jpg", ".bmp"])
        )
        self._connected: bool = False
        self._known_files: Set[str] = set()
        self._lock = threading.Lock()

        #: Path of the file that caused the most recent trigger.
        self.last_trigger_path: Optional[str] = None

    def connect(self) -> None:
        if not self._watch_dir.is_dir():
            raise ConnectionError(
                f"Watch directory does not exist: {self._watch_dir}"
            )
        # Snapshot existing files so only *new* arrivals trigger.
        with self._lock:
            self._known_files = self._scan_files()
        self._connected = True
        logger.info(
            "[FileWatch] Watching '%s' for %s files (%d existing ignored).",
            self._watch_dir, sorted(self._extensions), len(self._known_files),
        )

    def disconnect(self) -> None:
        self._connected = False
        logger.info("[FileWatch] Stopped watching '%s'.", self._watch_dir)

    @property
    def is_connected(self) -> bool:
        return self._connected

    def wait_for_trigger(self, timeout_ms: int = 5000) -> bool:
        """Poll for a new file in the watch directory.

        Returns True when a new file matching the configured extensions
        appears.  The file path is stored in :attr:`last_trigger_path`.
        """
        if not self._connected:
            logger.warning("[FileWatch] Not connected; cannot wait for trigger.")
            return False

        deadline = time.monotonic() + timeout_ms / 1000.0
        poll_interval = 0.1  # 100 ms

        while time.monotonic() < deadline:
            current = self._scan_files()
            with self._lock:
                new_files = current - self._known_files
                if new_files:
                    # Pick the alphabetically first new file for determinism.
                    picked = sorted(new_files)[0]
                    self._known_files = current
                    self.last_trigger_path = picked
                    logger.info(
                        "[FileWatch] Trigger: new file '%s'.",
                        os.path.basename(picked),
                    )
                    return True
            time.sleep(poll_interval)

        return False

    def send_result(self, is_pass: bool, score: float) -> None:
        label = "OK" if is_pass else "NG"
        filename = (
            os.path.basename(self.last_trigger_path)
            if self.last_trigger_path
            else "<unknown>"
        )
        logger.info(
            "[FileWatch] Result for '%s': %s  score=%.4f",
            filename, label, score,
        )

    def read_register(self, address: int) -> int:
        logger.debug(
            "[FileWatch] read_register(%d) -> 0 (not supported).", address,
        )
        return 0

    def write_register(self, address: int, value: int) -> None:
        logger.debug(
            "[FileWatch] write_register(%d, %d) (not supported).",
            address, value,
        )

    # -- internal helpers --------------------------------------------------

    def _scan_files(self) -> Set[str]:
        """Return the set of matching file paths currently in the directory."""
        try:
            return {
                str(p)
                for p in self._watch_dir.iterdir()
                if p.is_file() and p.suffix.lower() in self._extensions
            }
        except OSError:
            logger.warning(
                "[FileWatch] Failed to scan '%s'.", self._watch_dir,
                exc_info=True,
            )
            return set()
