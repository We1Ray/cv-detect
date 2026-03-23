"""
core/motion_interface.py - Motion control interface for vision-guided robotics.

Provides an abstract interface and concrete implementations for commanding
multi-axis motion controllers from a machine-vision pipeline.  Includes
coordinate-frame transforms using hand-eye calibration results, Modbus TCP
communication, and a simulated backend for offline development/testing.

Categories:
    1. Data Classes (AxisPosition, MotionConfig)
    2. Coordinate Frame Enum
    3. Abstract Motion Interface
    4. Modbus TCP Implementation
    5. Simulated Implementation
"""

from __future__ import annotations

import logging
import math
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ====================================================================== #
#  Named constants                                                        #
# ====================================================================== #

DEFAULT_TIMEOUT: float = 10.0          # seconds
DEFAULT_MODBUS_PORT: int = 502
MOTION_POLL_INTERVAL: float = 0.05     # seconds
DEFAULT_SIM_SPEED: float = 100.0       # units per second


# ====================================================================== #
#  Data Classes                                                           #
# ====================================================================== #


@dataclass
class AxisPosition:
    """Six-axis position representation (Cartesian + Euler angles)."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    rx: float = 0.0
    ry: float = 0.0
    rz: float = 0.0

    def to_array(self) -> np.ndarray:
        """Return a flat ``(6,)`` numpy array."""
        return np.array([self.x, self.y, self.z, self.rx, self.ry, self.rz], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> AxisPosition:
        """Construct from a ``(6,)`` array-like."""
        a = np.asarray(arr, dtype=np.float64).ravel()
        if a.shape[0] != 6:
            raise ValueError(f"Expected 6 elements, got {a.shape[0]}")
        return cls(x=a[0], y=a[1], z=a[2], rx=a[3], ry=a[4], rz=a[5])

    def distance_to(self, other: AxisPosition) -> float:
        """Euclidean distance in XYZ to *other*."""
        return float(np.linalg.norm(self.to_array()[:3] - other.to_array()[:3]))


@dataclass
class MotionConfig:
    """Connection and behaviour configuration for a motion controller."""
    protocol: str = "modbus_tcp"
    host: str = "127.0.0.1"
    port: int = DEFAULT_MODBUS_PORT
    timeout: float = DEFAULT_TIMEOUT
    axis_names: List[str] = field(default_factory=lambda: ["X", "Y", "Z", "RX", "RY", "RZ"])


# ====================================================================== #
#  Coordinate Frames                                                      #
# ====================================================================== #


class CoordinateFrame(Enum):
    """Reference frames for coordinate transforms."""
    WORLD = "world"
    CAMERA = "camera"
    ROBOT = "robot"
    TOOL = "tool"


# ====================================================================== #
#  Abstract Motion Interface                                              #
# ====================================================================== #


class MotionInterface(ABC):
    """Abstract base class for motion controller communication.

    All public methods are safe to call from any thread; concrete
    implementations should use appropriate locking where necessary.
    """

    def __init__(self, config: Optional[MotionConfig] = None) -> None:
        self.config = config or MotionConfig()
        self._transforms: Dict[Tuple[CoordinateFrame, CoordinateFrame], np.ndarray] = {}

    # ------------------------------------------------------------------ #
    #  Connection lifecycle                                               #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def connect(self) -> None:
        """Open a connection to the motion controller.

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
        """Return True if currently connected."""

    # ------------------------------------------------------------------ #
    #  Motion commands                                                    #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def get_position(self) -> AxisPosition:
        """Read the current position from the controller."""

    @abstractmethod
    def move_to(self, position: AxisPosition, speed: float = 50.0) -> None:
        """Command an absolute move to *position* at *speed* (units/s)."""

    @abstractmethod
    def move_relative(self, delta: AxisPosition, speed: float = 50.0) -> None:
        """Command a relative move by *delta* at *speed* (units/s)."""

    @abstractmethod
    def is_moving(self) -> bool:
        """Return True if any axis is currently in motion."""

    def wait_motion_complete(self, timeout: Optional[float] = None) -> bool:
        """Block until motion finishes or *timeout* elapses.

        Returns
        -------
        bool
            True if motion completed, False if timed out.
        """
        timeout = timeout or self.config.timeout
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self.is_moving():
                return True
            time.sleep(MOTION_POLL_INTERVAL)
        logger.warning("wait_motion_complete timed out after %.1fs", timeout)
        return False

    # ------------------------------------------------------------------ #
    #  Digital I/O                                                        #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def set_output(self, channel: int, value: bool) -> None:
        """Set a digital output *channel* to *value*."""

    @abstractmethod
    def get_input(self, channel: int) -> bool:
        """Read the state of a digital input *channel*."""

    # ------------------------------------------------------------------ #
    #  Coordinate frame transforms                                       #
    # ------------------------------------------------------------------ #

    def register_transform(
        self,
        from_frame: CoordinateFrame,
        to_frame: CoordinateFrame,
        matrix: np.ndarray,
    ) -> None:
        """Register a 4x4 homogeneous transform between two frames.

        Parameters
        ----------
        from_frame, to_frame:
            Source and destination coordinate frames.
        matrix:
            4x4 homogeneous transformation matrix.
        """
        mat = np.asarray(matrix, dtype=np.float64)
        if mat.shape != (4, 4):
            raise ValueError(f"Expected 4x4 matrix, got shape {mat.shape}")
        self._transforms[(from_frame, to_frame)] = mat
        # Store the inverse using efficient RT decomposition
        # (avoids numerical issues of general matrix inverse for rigid transforms)
        R = mat[:3, :3]
        t = mat[:3, 3]
        inv = np.eye(4, dtype=np.float64)
        inv[:3, :3] = R.T
        inv[:3, 3] = -R.T @ t
        self._transforms[(to_frame, from_frame)] = inv
        logger.info("Registered transform %s -> %s", from_frame.value, to_frame.value)

    def transform_to_frame(
        self,
        position: AxisPosition,
        from_frame: CoordinateFrame,
        to_frame: CoordinateFrame,
    ) -> AxisPosition:
        """Transform *position* from *from_frame* to *to_frame*.

        Uses registered 4x4 homogeneous matrices (typically from hand-eye
        calibration).  Both translational (XYZ) and rotational (RX, RY, RZ)
        components are transformed.
        """
        key = (from_frame, to_frame)
        if key not in self._transforms:
            raise ValueError(
                f"No transform registered for {from_frame.value} -> {to_frame.value}"
            )
        T = self._transforms[key]
        pt = np.array([position.x, position.y, position.z, 1.0], dtype=np.float64)
        transformed = T @ pt

        # Transform rotations via rotation matrix composition
        rx, ry, rz = position.rx, position.ry, position.rz
        try:
            from shared.core.hand_eye_calibration import (
                euler_to_rotation_matrix,
                rotation_matrix_to_euler,
            )
            R_transform = T[:3, :3]
            R_source = euler_to_rotation_matrix(rx, ry, rz)
            R_result = R_transform @ R_source
            rx, ry, rz = rotation_matrix_to_euler(R_result)
        except (ImportError, Exception):
            # Fallback: pass through rotations unchanged if hand_eye_calibration
            # is unavailable (e.g. missing cv2 dependency).
            pass

        return AxisPosition(
            x=transformed[0],
            y=transformed[1],
            z=transformed[2],
            rx=rx,
            ry=ry,
            rz=rz,
        )


# ====================================================================== #
#  Modbus TCP Implementation                                              #
# ====================================================================== #


class ModbusMotionInterface(MotionInterface):
    """Motion controller interface via Modbus TCP.

    Register layout (configurable via constructor):
        - Holding registers 0-5: current axis positions (X, Y, Z, RX, RY, RZ)
        - Holding registers 10-15: target axis positions
        - Holding register 20: commanded speed
        - Coil 0: motion trigger (write True to start move)
        - Coil 1: motion busy flag (read)
        - Coils 100+: digital outputs
        - Discrete inputs 100+: digital inputs

    Parameters
    ----------
    config:
        Connection configuration.
    position_scale:
        Scale factor applied when converting register values to floats
        (e.g. 100.0 means register value 12345 -> 123.45 mm).
    """

    def __init__(
        self,
        config: Optional[MotionConfig] = None,
        position_scale: float = 100.0,
        current_pos_start: int = 0,
        target_pos_start: int = 10,
        speed_register: int = 20,
        trigger_coil: int = 0,
        busy_coil: int = 1,
        io_coil_offset: int = 100,
    ) -> None:
        super().__init__(config)
        self._scale = position_scale
        self._reg_cur = current_pos_start
        self._reg_tgt = target_pos_start
        self._reg_speed = speed_register
        self._coil_trigger = trigger_coil
        self._coil_busy = busy_coil
        self._io_offset = io_coil_offset
        self._client: Any = None
        self._lock = threading.Lock()

    def connect(self) -> None:
        try:
            from pymodbus.client import ModbusTcpClient
        except ImportError:
            raise ImportError(
                "pymodbus is required for ModbusMotionInterface. "
                "Install it with: pip install pymodbus"
            )
        self._client = ModbusTcpClient(
            host=self.config.host,
            port=self.config.port,
            timeout=self.config.timeout,
        )
        if not self._client.connect():
            raise ConnectionError(
                f"Cannot connect to Modbus at {self.config.host}:{self.config.port}"
            )
        logger.info("Connected to Modbus at %s:%d", self.config.host, self.config.port)

    def disconnect(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
            logger.info("Modbus connection closed")

    @property
    def is_connected(self) -> bool:
        return self._client is not None and self._client.is_socket_open()

    def get_position(self) -> AxisPosition:
        self._ensure_connected()
        with self._lock:
            rr = self._client.read_holding_registers(self._reg_cur, 6)
            if rr.isError():
                raise IOError(f"Modbus read error: {rr}")
            vals = [float(v) / self._scale for v in rr.registers]
        return AxisPosition(*vals)

    def move_to(self, position: AxisPosition, speed: float = 50.0) -> None:
        self._ensure_connected()
        regs = [int(round(v * self._scale)) for v in position.to_array()]
        with self._lock:
            self._client.write_registers(self._reg_tgt, regs)
            self._client.write_register(self._reg_speed, int(round(speed * self._scale)))
            self._client.write_coil(self._coil_trigger, True)
        logger.debug("move_to %s at speed=%.1f", position, speed)

    def move_relative(self, delta: AxisPosition, speed: float = 50.0) -> None:
        current = self.get_position()
        target = AxisPosition.from_array(current.to_array() + delta.to_array())
        self.move_to(target, speed)

    def is_moving(self) -> bool:
        self._ensure_connected()
        with self._lock:
            rr = self._client.read_coils(self._coil_busy, 1)
            if rr.isError():
                raise IOError(f"Modbus read error: {rr}")
            return bool(rr.bits[0])

    def set_output(self, channel: int, value: bool) -> None:
        self._ensure_connected()
        with self._lock:
            self._client.write_coil(self._io_offset + channel, value)

    def get_input(self, channel: int) -> bool:
        self._ensure_connected()
        with self._lock:
            rr = self._client.read_discrete_inputs(self._io_offset + channel, 1)
            if rr.isError():
                raise IOError(f"Modbus read error: {rr}")
            return bool(rr.bits[0])

    def _ensure_connected(self) -> None:
        if not self.is_connected:
            raise ConnectionError("Modbus client is not connected")


# ====================================================================== #
#  Simulated Implementation                                               #
# ====================================================================== #


class SimulatedMotionInterface(MotionInterface):
    """In-memory simulated motion controller for testing and development.

    Parameters
    ----------
    config:
        Configuration (protocol field is ignored).
    sim_speed:
        Simulated motion speed in units per second.
    """

    def __init__(
        self,
        config: Optional[MotionConfig] = None,
        sim_speed: float = DEFAULT_SIM_SPEED,
    ) -> None:
        super().__init__(config)
        self._sim_speed = sim_speed
        self._connected = False
        self._position = AxisPosition()
        self._target: Optional[AxisPosition] = None
        self._moving = False
        self._outputs: Dict[int, bool] = {}
        self._inputs: Dict[int, bool] = {}
        self._command_log: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._motion_thread: Optional[threading.Thread] = None
        self._cancel_motion = False

    # ------------------------------------------------------------------ #
    #  Connection                                                         #
    # ------------------------------------------------------------------ #

    def connect(self) -> None:
        self._connected = True
        self._log_cmd("connect")
        logger.info("Simulated motion controller connected")

    def disconnect(self) -> None:
        self._connected = False
        self._moving = False
        self._log_cmd("disconnect")
        logger.info("Simulated motion controller disconnected")

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------ #
    #  Motion                                                             #
    # ------------------------------------------------------------------ #

    def get_position(self) -> AxisPosition:
        self._ensure_connected()
        with self._lock:
            return AxisPosition.from_array(self._position.to_array().copy())

    def move_to(self, position: AxisPosition, speed: float = 50.0) -> None:
        self._ensure_connected()
        self._log_cmd("move_to", position=position, speed=speed)

        # Cancel any existing motion thread before starting a new one
        self._cancel_motion = True
        if self._motion_thread and self._motion_thread.is_alive():
            self._motion_thread.join(timeout=1.0)
        self._cancel_motion = False

        with self._lock:
            self._target = position
            self._moving = True

        # Simulate motion in a single background thread
        self._motion_thread = threading.Thread(
            target=self._simulate_motion, args=(position, speed), daemon=True,
        )
        self._motion_thread.start()

    def move_relative(self, delta: AxisPosition, speed: float = 50.0) -> None:
        current = self.get_position()
        target = AxisPosition.from_array(current.to_array() + delta.to_array())
        self._log_cmd("move_relative", delta=delta, speed=speed)
        self.move_to(target, speed)

    def is_moving(self) -> bool:
        return self._moving

    # ------------------------------------------------------------------ #
    #  Digital I/O                                                        #
    # ------------------------------------------------------------------ #

    def set_output(self, channel: int, value: bool) -> None:
        self._ensure_connected()
        self._outputs[channel] = value
        self._log_cmd("set_output", channel=channel, value=value)

    def get_input(self, channel: int) -> bool:
        self._ensure_connected()
        return self._inputs.get(channel, False)

    def set_simulated_input(self, channel: int, value: bool) -> None:
        """Inject a simulated digital input value (for testing)."""
        self._inputs[channel] = value

    # ------------------------------------------------------------------ #
    #  Introspection                                                      #
    # ------------------------------------------------------------------ #

    @property
    def command_log(self) -> List[Dict[str, Any]]:
        """Return the full log of commands issued to this interface."""
        return list(self._command_log)

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _simulate_motion(self, target: AxisPosition, speed: float) -> None:
        """Interpolate from current position to *target* at *speed*."""
        start = self._position.to_array()
        end = target.to_array()
        distance = float(np.linalg.norm(end[:3] - start[:3]))
        if distance < 1e-6:
            with self._lock:
                self._position = target
                self._moving = False
            return

        duration = distance / max(speed, 1e-6)
        t0 = time.monotonic()

        while True:
            if self._cancel_motion:
                logger.debug("Simulated motion cancelled")
                return
            elapsed = time.monotonic() - t0
            ratio = min(elapsed / duration, 1.0)
            interp = start + (end - start) * ratio
            with self._lock:
                self._position = AxisPosition.from_array(interp)
            if ratio >= 1.0:
                break
            time.sleep(MOTION_POLL_INTERVAL)

        with self._lock:
            self._position = target
            self._moving = False
        logger.debug("Simulated motion complete -> %s", target)

    def _log_cmd(self, name: str, **kwargs: Any) -> None:
        entry: Dict[str, Any] = {"command": name, "timestamp": time.time()}
        for key, val in kwargs.items():
            if isinstance(val, AxisPosition):
                entry[key] = val.to_array().tolist()
            else:
                entry[key] = val
        self._command_log.append(entry)
        logger.debug("SIM CMD: %s %s", name, kwargs)

    def _ensure_connected(self) -> None:
        if not self._connected:
            raise ConnectionError("Simulated controller is not connected")
