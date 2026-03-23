"""GenICam/GigE Vision complete interface for industrial camera control.

Provides a high-level Python interface to GenICam-compliant cameras via the
``harvesters`` library (GenTL consumer).  When ``harvesters`` is not installed,
a stub implementation is provided that logs warnings so that dependent code
can still be imported and tested without hardware.

Supported transports:
    - GigE Vision (via GenTL producer / .cti file)
    - USB3 Vision (via GenTL producer)
    - CoaXPress (if producer is available)

Usage::

    mgr = GenICamManager(cti_path="/opt/pylon/lib/pylonCXP/pylon.cti")
    devices = mgr.discover_devices()
    mgr.connect(devices[0].serial)
    mgr.set_exposure(5000)
    frame = mgr.grab_frame()
    mgr.disconnect()
"""

from __future__ import annotations

import logging
import os
import platform
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GenTL producer (.cti) common search paths
# ---------------------------------------------------------------------------
_CTI_SEARCH_PATHS: List[str] = []
if platform.system() == "Linux":
    _CTI_SEARCH_PATHS = [
        "/opt/mvIMPACT_Acquire/lib/x86_64",
        "/opt/pylon/lib/pylonCXP",
        "/opt/ids-peak/lib",
        "/usr/lib",
    ]
elif platform.system() == "Darwin":
    _CTI_SEARCH_PATHS = [
        "/Library/Frameworks/pylon.framework/Libraries",
        "/opt/mvIMPACT_Acquire/lib",
    ]
elif platform.system() == "Windows":
    _CTI_SEARCH_PATHS = [
        r"C:\Program Files\Basler\pylon 7\Runtime\x64",
        r"C:\Program Files\MATRIX VISION\mvIMPACT Acquire\bin\x64",
        r"C:\Program Files\Allied Vision\Vimba X\GenTL",
    ]

# Attempt to import harvesters; set flag for fallback behaviour.
try:
    from harvesters.core import Harvester, ImageAcquirer, Buffer

    _HAS_HARVESTERS = True
except ImportError:
    _HAS_HARVESTERS = False
    logger.info(
        "harvesters library not found. GenICam interface will use stub mode. "
        "Install with: pip install harvesters"
    )


# =========================================================================
# Feature type enumeration
# =========================================================================
class FeatureType(str, Enum):
    """GenICam feature node types."""

    INTEGER = "Integer"
    FLOAT = "Float"
    BOOLEAN = "Boolean"
    ENUM = "Enumeration"
    STRING = "String"
    COMMAND = "Command"
    UNKNOWN = "Unknown"


# =========================================================================
# Data classes
# =========================================================================
@dataclass
class GenICamDevice:
    """Discovered GenICam device descriptor.

    Attributes
    ----------
    serial : str
        Unique serial number of the camera.
    model : str
        Camera model name.
    vendor : str
        Manufacturer / vendor name.
    ip : str
        IP address (GigE devices) or empty string.
    mac : str
        MAC address or empty string.
    features : Dict[str, FeatureType]
        Map of available feature names to their node types.  Populated
        after connection (empty on discovery).
    """

    serial: str
    model: str
    vendor: str
    ip: str = ""
    mac: str = ""
    features: Dict[str, FeatureType] = field(default_factory=dict)


@dataclass
class FeatureRange:
    """Numeric feature range descriptor.

    Attributes
    ----------
    min_val : float
        Minimum allowed value.
    max_val : float
        Maximum allowed value.
    increment : float
        Step size (0 if continuous).
    current : float
        Current value at time of query.
    """

    min_val: float
    max_val: float
    increment: float
    current: float


@dataclass
class FeatureInfo:
    """Full description of a single GenICam feature node.

    Attributes
    ----------
    name : str
        Feature name (e.g. ``"ExposureTime"``).
    feature_type : FeatureType
        Node type.
    value : Any
        Current value (type depends on *feature_type*).
    writable : bool
        Whether the feature is currently writable.
    description : str
        Human-readable description from the device XML.
    """

    name: str
    feature_type: FeatureType
    value: Any = None
    writable: bool = True
    description: str = ""


# =========================================================================
# GenICamManager
# =========================================================================
class GenICamManager:
    """High-level manager for GenICam device discovery and operation.

    Parameters
    ----------
    cti_path : str, optional
        Path to a GenTL producer (``.cti`` file).  If not provided, the
        manager attempts to locate one automatically from well-known paths
        and the ``GENICAM_GENTL64_PATH`` / ``GENICAM_GENTL32_PATH``
        environment variables.
    """

    def __init__(self, cti_path: Optional[str] = None) -> None:
        self._cti_path = cti_path or self._find_cti()
        self._harvester: Optional[Any] = None  # Harvester instance
        self._acquirer: Optional[Any] = None  # ImageAcquirer instance
        self._connected_device: Optional[GenICamDevice] = None
        self._is_acquiring = False
        self._lock = threading.Lock()

        if _HAS_HARVESTERS and self._cti_path:
            self._harvester = Harvester()
            self._harvester.add_file(self._cti_path)
            logger.info("GenICamManager initialised with CTI: %s", self._cti_path)
        elif _HAS_HARVESTERS and not self._cti_path:
            self._harvester = Harvester()
            logger.warning(
                "No .cti file found. Call add_cti_file() before discover_devices()."
            )
        else:
            logger.warning(
                "harvesters not installed -- GenICamManager running in stub mode."
            )

    # ------------------------------------------------------------------
    # CTI file management
    # ------------------------------------------------------------------

    def add_cti_file(self, path: str) -> None:
        """Add a GenTL producer file path.

        Parameters
        ----------
        path : str
            Absolute path to a ``.cti`` file.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        RuntimeError
            If harvesters is not installed.
        """
        if not _HAS_HARVESTERS:
            raise RuntimeError("harvesters library is not installed.")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"CTI file not found: {path}")
        self._cti_path = path
        self._harvester.add_file(path)
        logger.info("Added CTI file: %s", path)

    # ------------------------------------------------------------------
    # Device discovery
    # ------------------------------------------------------------------

    def discover_devices(self, timeout_s: float = 3.0) -> List[GenICamDevice]:
        """Discover all GenICam devices on the network.

        Parameters
        ----------
        timeout_s : float
            Discovery timeout in seconds.

        Returns
        -------
        List[GenICamDevice]
            List of discovered devices.
        """
        if not _HAS_HARVESTERS or self._harvester is None:
            logger.warning("Stub mode: returning empty device list.")
            return []

        self._harvester.update()
        time.sleep(min(timeout_s, 1.0))

        devices: List[GenICamDevice] = []
        for dev_info in self._harvester.device_info_list:
            device = GenICamDevice(
                serial=getattr(dev_info, "serial_number", "") or "",
                model=getattr(dev_info, "model", "") or "",
                vendor=getattr(dev_info, "vendor", "") or "",
                ip=getattr(dev_info, "property_dict", {}).get(
                    "current_ip_address", ""
                ),
                mac=getattr(dev_info, "property_dict", {}).get(
                    "mac_address", ""
                ),
            )
            devices.append(device)
            logger.debug(
                "Discovered: %s %s (SN: %s, IP: %s)",
                device.vendor,
                device.model,
                device.serial,
                device.ip,
            )

        logger.info("Discovered %d GenICam device(s).", len(devices))
        return devices

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self, serial_or_ip: str) -> GenICamDevice:
        """Connect to a specific GenICam device.

        Parameters
        ----------
        serial_or_ip : str
            Serial number or IP address of the target device.

        Returns
        -------
        GenICamDevice
            Connected device descriptor with features populated.

        Raises
        ------
        ConnectionError
            If the device cannot be found or connection fails.
        """
        with self._lock:
            if self._acquirer is not None:
                logger.warning("Already connected. Disconnecting current device first.")
                self._disconnect_internal()

            if not _HAS_HARVESTERS or self._harvester is None:
                raise ConnectionError(
                    "Cannot connect: harvesters not installed or no CTI file loaded."
                )

            self._harvester.update()

            # Find the device by serial or IP.
            target_idx: Optional[int] = None
            for idx, dev_info in enumerate(self._harvester.device_info_list):
                serial = getattr(dev_info, "serial_number", "") or ""
                ip = getattr(dev_info, "property_dict", {}).get(
                    "current_ip_address", ""
                )
                if serial == serial_or_ip or ip == serial_or_ip:
                    target_idx = idx
                    break

            if target_idx is None:
                raise ConnectionError(
                    f"Device not found: {serial_or_ip}. "
                    f"Available devices: {len(self._harvester.device_info_list)}"
                )

            try:
                self._acquirer = self._harvester.create(target_idx)
                dev_info = self._harvester.device_info_list[target_idx]

                self._connected_device = GenICamDevice(
                    serial=getattr(dev_info, "serial_number", "") or "",
                    model=getattr(dev_info, "model", "") or "",
                    vendor=getattr(dev_info, "vendor", "") or "",
                )

                # Populate feature map from the remote device node map.
                self._connected_device.features = self._enumerate_features()

                logger.info(
                    "Connected to %s %s (SN: %s) -- %d features",
                    self._connected_device.vendor,
                    self._connected_device.model,
                    self._connected_device.serial,
                    len(self._connected_device.features),
                )
                return self._connected_device

            except Exception as exc:
                self._acquirer = None
                raise ConnectionError(
                    f"Failed to connect to {serial_or_ip}: {exc}"
                ) from exc

    def disconnect(self) -> None:
        """Disconnect from the current device and release resources."""
        with self._lock:
            self._disconnect_internal()

    def _disconnect_internal(self) -> None:
        """Internal disconnect (caller must hold ``_lock``)."""
        if self._is_acquiring:
            try:
                self._acquirer.stop()
            except Exception:
                pass
            self._is_acquiring = False

        if self._acquirer is not None:
            try:
                self._acquirer.destroy()
            except Exception:
                pass
            self._acquirer = None

        self._connected_device = None
        logger.info("Disconnected from device.")

    @property
    def is_connected(self) -> bool:
        """Return True if a device is currently connected."""
        return self._acquirer is not None and self._connected_device is not None

    # ------------------------------------------------------------------
    # Feature access
    # ------------------------------------------------------------------

    def get_feature(self, name: str) -> Any:
        """Read a GenICam feature value.

        Parameters
        ----------
        name : str
            Feature name (e.g. ``"ExposureTime"``, ``"Gain"``,
            ``"TriggerMode"``).

        Returns
        -------
        Any
            Current value.  Type depends on the feature node type
            (int, float, bool, str).

        Raises
        ------
        RuntimeError
            If not connected.
        AttributeError
            If the feature does not exist.
        """
        node_map = self._require_node_map()
        try:
            node = getattr(node_map, name)
            return node.value
        except AttributeError:
            raise AttributeError(
                f"Feature '{name}' not found on device "
                f"{self._connected_device.model}."
            )

    def set_feature(self, name: str, value: Any) -> None:
        """Write a GenICam feature value.

        Parameters
        ----------
        name : str
            Feature name.
        value : Any
            Value to write.  Must match the feature node type.

        Raises
        ------
        RuntimeError
            If not connected or feature is not writable.
        """
        node_map = self._require_node_map()
        try:
            node = getattr(node_map, name)
            node.value = value
            logger.debug("Set feature %s = %s", name, value)
        except AttributeError:
            raise AttributeError(f"Feature '{name}' not found.")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to set '{name}' to {value!r}: {exc}"
            ) from exc

    def get_feature_range(self, name: str) -> FeatureRange:
        """Get the valid range of a numeric feature.

        Parameters
        ----------
        name : str
            Feature name (must be Integer or Float type).

        Returns
        -------
        FeatureRange
            Range descriptor with min, max, increment, and current value.
        """
        node_map = self._require_node_map()
        try:
            node = getattr(node_map, name)
            return FeatureRange(
                min_val=float(node.min),
                max_val=float(node.max),
                increment=float(getattr(node, "inc", 0)),
                current=float(node.value),
            )
        except AttributeError:
            raise AttributeError(f"Feature '{name}' not found.")

    def list_features(self) -> List[FeatureInfo]:
        """List all available features with their current values.

        Returns
        -------
        List[FeatureInfo]
            Feature descriptors sorted by name.
        """
        if not self.is_connected:
            logger.warning("Not connected -- returning empty feature list.")
            return []

        node_map = self._require_node_map()
        features: List[FeatureInfo] = []

        for name, feat_type in sorted(self._connected_device.features.items()):
            try:
                node = getattr(node_map, name)
                value = node.value if hasattr(node, "value") else None
                writable = getattr(node, "is_writable", True)
                desc = getattr(node, "description", "")
            except Exception:
                value = None
                writable = False
                desc = ""

            features.append(
                FeatureInfo(
                    name=name,
                    feature_type=feat_type,
                    value=value,
                    writable=writable,
                    description=desc,
                )
            )

        return features

    # ------------------------------------------------------------------
    # Acquisition
    # ------------------------------------------------------------------

    def grab_frame(self, timeout_s: float = 5.0) -> Optional[np.ndarray]:
        """Acquire a single frame from the connected camera.

        If continuous acquisition is not running, the method starts it
        temporarily and stops after grabbing one frame.

        Parameters
        ----------
        timeout_s : float
            Maximum seconds to wait for a frame.

        Returns
        -------
        np.ndarray or None
            Grabbed image (HxW or HxWxC ``uint8``), or ``None`` on timeout.
        """
        if not _HAS_HARVESTERS or self._acquirer is None:
            logger.warning("Stub mode: returning synthetic test frame.")
            return np.zeros((480, 640), dtype=np.uint8)

        was_acquiring = self._is_acquiring
        if not was_acquiring:
            self._acquirer.start()

        try:
            buffer = self._acquirer.fetch(timeout=timeout_s)
            if buffer is None:
                logger.warning("grab_frame timed out after %.1fs", timeout_s)
                return None

            component = buffer.payload.components[0]
            image = self._component_to_ndarray(component)
            buffer.queue()
            return image

        except Exception as exc:
            logger.error("grab_frame failed: %s", exc, exc_info=True)
            return None

        finally:
            if not was_acquiring:
                self._acquirer.stop()

    def start_acquisition(self) -> None:
        """Start continuous frame acquisition.

        Raises
        ------
        RuntimeError
            If not connected or already acquiring.
        """
        if not _HAS_HARVESTERS or self._acquirer is None:
            logger.warning("Stub mode: start_acquisition is a no-op.")
            self._is_acquiring = True
            return

        if self._is_acquiring:
            logger.warning("Acquisition already running.")
            return

        self._acquirer.start()
        self._is_acquiring = True
        logger.info("Continuous acquisition started.")

    def stop_acquisition(self) -> None:
        """Stop continuous frame acquisition."""
        if not self._is_acquiring:
            return

        if _HAS_HARVESTERS and self._acquirer is not None:
            self._acquirer.stop()

        self._is_acquiring = False
        logger.info("Continuous acquisition stopped.")

    # ------------------------------------------------------------------
    # Trigger configuration
    # ------------------------------------------------------------------

    def set_trigger_mode(
        self,
        source: str = "Line1",
        activation: str = "RisingEdge",
    ) -> None:
        """Configure hardware trigger mode.

        Parameters
        ----------
        source : str
            Trigger source (e.g. ``"Line1"``, ``"Line2"``, ``"Software"``).
        activation : str
            Edge type (``"RisingEdge"``, ``"FallingEdge"``, ``"AnyEdge"``).
        """
        if not self.is_connected:
            logger.warning("Stub mode: set_trigger_mode is a no-op.")
            return

        try:
            self.set_feature("TriggerMode", "On")
            self.set_feature("TriggerSource", source)
            self.set_feature("TriggerActivation", activation)
            logger.info(
                "Trigger configured: source=%s, activation=%s", source, activation
            )
        except Exception as exc:
            logger.error("Failed to set trigger mode: %s", exc)
            raise

    def send_software_trigger(self) -> None:
        """Issue a software trigger command.

        Requires ``TriggerSource`` set to ``"Software"``.
        """
        if not self.is_connected:
            logger.warning("Stub mode: send_software_trigger is a no-op.")
            return

        node_map = self._require_node_map()
        try:
            node_map.TriggerSoftware.execute()
            logger.debug("Software trigger sent.")
        except Exception as exc:
            logger.error("Software trigger failed: %s", exc)

    # ------------------------------------------------------------------
    # Common parameter shortcuts
    # ------------------------------------------------------------------

    def set_exposure(self, microseconds: float) -> None:
        """Set exposure time in microseconds.

        Parameters
        ----------
        microseconds : float
            Exposure time (must be positive).
        """
        if microseconds <= 0:
            raise ValueError("Exposure time must be positive.")
        self.set_feature("ExposureTime", float(microseconds))
        logger.info("Exposure set to %.1f us", microseconds)

    def set_gain(self, db: float) -> None:
        """Set analog gain in dB.

        Parameters
        ----------
        db : float
            Gain value in decibels (must be non-negative).
        """
        if db < 0:
            raise ValueError("Gain must be non-negative.")
        self.set_feature("Gain", float(db))
        logger.info("Gain set to %.2f dB", db)

    def set_roi(
        self,
        offset_x: int,
        offset_y: int,
        width: int,
        height: int,
    ) -> None:
        """Set the region of interest (ROI) on the sensor.

        Parameters
        ----------
        offset_x, offset_y : int
            Top-left corner of the ROI.
        width, height : int
            ROI dimensions in pixels.
        """
        self.set_feature("OffsetX", offset_x)
        self.set_feature("OffsetY", offset_y)
        self.set_feature("Width", width)
        self.set_feature("Height", height)
        logger.info(
            "ROI set to (%d, %d) %dx%d", offset_x, offset_y, width, height
        )

    # ------------------------------------------------------------------
    # User set management
    # ------------------------------------------------------------------

    def save_user_set(self, user_set: str = "UserSet1") -> None:
        """Save current camera configuration to a user set.

        Parameters
        ----------
        user_set : str
            User set name (e.g. ``"UserSet1"``, ``"UserSet2"``).
        """
        if not self.is_connected:
            logger.warning("Stub mode: save_user_set is a no-op.")
            return

        try:
            self.set_feature("UserSetSelector", user_set)
            node_map = self._require_node_map()
            node_map.UserSetSave.execute()
            logger.info("Configuration saved to %s.", user_set)
        except Exception as exc:
            logger.error("Failed to save user set '%s': %s", user_set, exc)
            raise

    def load_user_set(self, user_set: str = "UserSet1") -> None:
        """Load a previously saved camera configuration.

        Parameters
        ----------
        user_set : str
            User set name to load.
        """
        if not self.is_connected:
            logger.warning("Stub mode: load_user_set is a no-op.")
            return

        try:
            self.set_feature("UserSetSelector", user_set)
            node_map = self._require_node_map()
            node_map.UserSetLoad.execute()
            logger.info("Configuration loaded from %s.", user_set)
        except Exception as exc:
            logger.error("Failed to load user set '%s': %s", user_set, exc)
            raise

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Disconnect and release all resources."""
        self.disconnect()
        if _HAS_HARVESTERS and self._harvester is not None:
            self._harvester.reset()
            logger.info("Harvester reset complete.")

    def __enter__(self) -> GenICamManager:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.shutdown()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_node_map(self) -> Any:
        """Return the remote device node map, or raise if not connected."""
        if self._acquirer is None:
            raise RuntimeError("Not connected to any device.")
        return self._acquirer.remote_device.node_map

    def _enumerate_features(self) -> Dict[str, FeatureType]:
        """Walk the node map and return a name -> type mapping."""
        features: Dict[str, FeatureType] = {}
        if self._acquirer is None:
            return features

        try:
            node_map = self._acquirer.remote_device.node_map
            for node in node_map._nodes:
                name = node.name if hasattr(node, "name") else str(node)
                interface = getattr(node, "interface_type", None)
                feat_type = _INTERFACE_TO_TYPE.get(interface, FeatureType.UNKNOWN)
                features[name] = feat_type
        except Exception as exc:
            logger.debug("Feature enumeration failed: %s", exc)

        return features

    @staticmethod
    def _component_to_ndarray(component: Any) -> np.ndarray:
        """Convert a harvesters buffer component to a numpy array."""
        width = component.width
        height = component.height
        data_format = component.data_format

        raw = component.data.copy()

        # Handle mono formats.
        if "Mono8" in str(data_format):
            return raw.reshape((height, width))
        if "Mono10" in str(data_format) or "Mono12" in str(data_format):
            arr = raw.view(np.uint16).reshape((height, width))
            return (arr >> 4).astype(np.uint8)
        if "Mono16" in str(data_format):
            arr = raw.view(np.uint16).reshape((height, width))
            return (arr >> 8).astype(np.uint8)

        # Handle colour formats (Bayer / RGB / BGR).
        if "BayerRG8" in str(data_format):
            import cv2

            bayer = raw.reshape((height, width))
            return cv2.cvtColor(bayer, cv2.COLOR_BayerRG2BGR)
        if "BayerGR8" in str(data_format):
            import cv2

            bayer = raw.reshape((height, width))
            return cv2.cvtColor(bayer, cv2.COLOR_BayerGR2BGR)
        if "RGB8" in str(data_format):
            return raw.reshape((height, width, 3))[:, :, ::-1]  # RGB -> BGR
        if "BGR8" in str(data_format):
            return raw.reshape((height, width, 3))

        # Fallback: try to reshape as mono.
        logger.warning("Unknown pixel format '%s' -- treating as Mono8.", data_format)
        return raw.reshape((height, width))

    @staticmethod
    def _find_cti() -> Optional[str]:
        """Attempt to auto-locate a .cti file."""
        # Check environment variables first.
        for env_var in ("GENICAM_GENTL64_PATH", "GENICAM_GENTL32_PATH"):
            paths = os.environ.get(env_var, "")
            for p in paths.split(os.pathsep):
                p = p.strip()
                if not p:
                    continue
                for entry in _safe_listdir(p):
                    if entry.endswith(".cti"):
                        full = os.path.join(p, entry)
                        logger.info("Auto-detected CTI from %s: %s", env_var, full)
                        return full

        # Search well-known paths.
        for search_dir in _CTI_SEARCH_PATHS:
            for entry in _safe_listdir(search_dir):
                if entry.endswith(".cti"):
                    full = os.path.join(search_dir, entry)
                    logger.info("Auto-detected CTI: %s", full)
                    return full

        logger.debug("No .cti file found automatically.")
        return None


# =========================================================================
# Module-level helpers
# =========================================================================

_INTERFACE_TO_TYPE: Dict[Any, FeatureType] = {}

# Populate the interface-to-type mapping if harvesters is available.
try:
    from genicam.genapi import EInterfaceType

    _INTERFACE_TO_TYPE = {
        EInterfaceType.intfIInteger: FeatureType.INTEGER,
        EInterfaceType.intfIFloat: FeatureType.FLOAT,
        EInterfaceType.intfIBoolean: FeatureType.BOOLEAN,
        EInterfaceType.intfIEnumeration: FeatureType.ENUM,
        EInterfaceType.intfIString: FeatureType.STRING,
        EInterfaceType.intfICommand: FeatureType.COMMAND,
    }
except ImportError:
    pass


def _safe_listdir(path: str) -> List[str]:
    """List directory contents, returning empty list on error."""
    try:
        return os.listdir(path)
    except OSError:
        return []
