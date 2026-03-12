"""Industrial camera acquisition module.

Provides a unified interface for industrial cameras (GigE Vision, USB3 Vision)
and regular USB webcams as fallback.  All heavy imports (harvesters, vimba, etc.)
are lazy-loaded with graceful fallback to OpenCV.

Supported backends:
    - GenTL (harvesters): GigE Vision, USB3 Vision via GenICam/GenTL producers
    - OpenCV: USB webcams, RTSP streams (always available)

Usage::

    mgr = CameraManager()
    cameras = mgr.discover_all()
    mgr.open(cameras[0].id)
    frame = mgr.grab_single()
    mgr.close()
"""

from __future__ import annotations

import logging
import os
import platform
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from shared.op_logger import log_operation
from shared.validation import validate_image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 常數定義
# ---------------------------------------------------------------------------
_PROBE_MAX_INDEX = 5  # OpenCV 探測 webcam 最大 index
_GRAB_TIMEOUT_S = 5.0  # 單張擷取逾時 (秒)
_STREAM_POLL_INTERVAL = 0.001  # 串流迴圈輪詢間隔 (秒)

# GenTL producer (.cti) 常見搜尋路徑
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
        r"C:\Program Files\IDS\ids_peak\comfort_backend\bin",
    ]


# =========================================================================
# 資料類別
# =========================================================================
@dataclass
class CameraInfo:
    """Discovered camera information."""

    id: str  # 唯一識別碼
    name: str  # 顯示名稱
    vendor: str  # 製造商
    model: str  # 型號
    serial: str  # 序號
    interface: str  # "GigE", "USB3", "USB", "GenTL", "Webcam"
    is_available: bool = True

    def __str__(self) -> str:
        return f"[{self.interface}] {self.vendor} {self.model} (SN: {self.serial})"


@dataclass
class FrameResult:
    """A single acquired frame."""

    image: np.ndarray  # BGR 或灰階 uint8
    timestamp: float  # 擷取時間戳 (time.time())
    frame_id: int  # 循序影格編號
    width: int
    height: int
    exposure_us: float = 0.0  # 曝光時間 (微秒)
    gain_db: float = 0.0  # 增益 (dB)


# =========================================================================
# 影格格式轉換工具
# =========================================================================

def _convert_pixel_format(
    buffer_array: np.ndarray,
    pixel_format: str,
    width: int,
    height: int,
) -> np.ndarray:
    """將原始像素格式轉換為 BGR 或灰階 uint8 numpy 陣列.

    支援格式:
        - Mono8: 直接使用
        - Mono10 / Mono12: 右移至 8-bit
        - BayerRG8: OpenCV demosaic
        - RGB8 / BGR8: 色彩空間轉換
    """
    import cv2

    fmt = pixel_format.lower() if pixel_format else ""

    # --- Mono8 ---
    if "mono8" in fmt or fmt == "mono8":
        return buffer_array.reshape((height, width)).astype(np.uint8)

    # --- Mono10 / Mono12 (高位元深度灰階) ---
    if "mono10" in fmt or "mono12" in fmt:
        arr = buffer_array.reshape((height, width))
        shift = 2 if "mono10" in fmt else 4
        return (arr >> shift).astype(np.uint8)

    # --- Mono16 ---
    if "mono16" in fmt:
        arr = buffer_array.reshape((height, width))
        return (arr >> 8).astype(np.uint8)

    # --- BayerRG8 ---
    if "bayerrg" in fmt:
        mono = buffer_array.reshape((height, width)).astype(np.uint8)
        return cv2.cvtColor(mono, cv2.COLOR_BayerRG2BGR)

    # --- BayerGB8 ---
    if "bayergb" in fmt:
        mono = buffer_array.reshape((height, width)).astype(np.uint8)
        return cv2.cvtColor(mono, cv2.COLOR_BayerGB2BGR)

    # --- BayerGR8 ---
    if "bayergr" in fmt:
        mono = buffer_array.reshape((height, width)).astype(np.uint8)
        return cv2.cvtColor(mono, cv2.COLOR_BayerGR2BGR)

    # --- BayerBG8 ---
    if "bayerbg" in fmt:
        mono = buffer_array.reshape((height, width)).astype(np.uint8)
        return cv2.cvtColor(mono, cv2.COLOR_BayerBG2BGR)

    # --- RGB8 ---
    if "rgb8" in fmt or "rgb" in fmt:
        arr = buffer_array.reshape((height, width, 3)).astype(np.uint8)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    # --- BGR8 (已經是目標格式) ---
    if "bgr8" in fmt or "bgr" in fmt:
        return buffer_array.reshape((height, width, 3)).astype(np.uint8)

    # --- 未知格式: 嘗試灰階 ---
    logger.warning("未知像素格式 '%s'，嘗試以 Mono8 處理", pixel_format)
    try:
        return buffer_array.reshape((height, width)).astype(np.uint8)
    except ValueError:
        return buffer_array.reshape((height, width, 3)).astype(np.uint8)


# =========================================================================
# 抽象後端基底類別
# =========================================================================

class CameraBackend(ABC):
    """Abstract base for camera backends."""

    @abstractmethod
    def discover(self) -> List[CameraInfo]:
        """探索並列舉可用相機."""
        ...

    @abstractmethod
    def open(self, camera_id: str) -> None:
        """開啟指定相機."""
        ...

    @abstractmethod
    def close(self) -> None:
        """關閉相機並釋放資源."""
        ...

    @abstractmethod
    def grab_single(self) -> Optional[FrameResult]:
        """擷取單張影格."""
        ...

    @abstractmethod
    def start_streaming(self, callback: Callable[[FrameResult], None]) -> None:
        """啟動連續串流, 每張影格呼叫 callback."""
        ...

    @abstractmethod
    def stop_streaming(self) -> None:
        """停止串流."""
        ...

    @abstractmethod
    def set_exposure(self, us: float) -> None:
        """設定曝光時間 (微秒)."""
        ...

    @abstractmethod
    def set_gain(self, db: float) -> None:
        """設定增益 (dB)."""
        ...

    @abstractmethod
    def get_exposure_range(self) -> Tuple[float, float]:
        """取得曝光時間範圍 (min_us, max_us)."""
        ...

    @abstractmethod
    def get_gain_range(self) -> Tuple[float, float]:
        """取得增益範圍 (min_db, max_db)."""
        ...

    @property
    @abstractmethod
    def is_open(self) -> bool:
        """相機是否已開啟."""
        ...

    @property
    @abstractmethod
    def is_streaming(self) -> bool:
        """是否正在串流."""
        ...


# =========================================================================
# GenTL 後端 (harvesters)
# =========================================================================

class GenTLBackend(CameraBackend):
    """GenICam/GenTL backend using harvesters library.

    Supports GigE Vision and USB3 Vision cameras through GenTL producers
    (.cti files). Searches common CTI paths and GENICAM_GENTL64_PATH env var.
    """

    def __init__(self) -> None:
        self._harvester = None  # lazy: harvesters.core.Harvester
        self._acquirer = None
        self._streaming = False
        self._stream_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._frame_count = 0
        self._current_camera: Optional[CameraInfo] = None
        self._lock = threading.Lock()
        self._available = False  # harvesters 是否可用

    # ----- 內部工具 -----

    def _ensure_harvester(self) -> bool:
        """延遲載入 harvesters 並初始化 Harvester 實例."""
        if self._harvester is not None:
            return True
        try:
            from harvesters.core import Harvester  # type: ignore[import-untyped]

            self._harvester = Harvester()
            self._load_cti_files()
            self._harvester.update()
            self._available = True
            logger.info("GenTL Harvester 初始化成功, 載入 %d 個裝置",
                        len(self._harvester.device_info_list))
            return True
        except ImportError:
            logger.debug("harvesters 套件未安裝, GenTL 後端不可用")
            return False
        except Exception as exc:  # noqa: BLE001
            logger.warning("GenTL Harvester 初始化失敗: %s", exc)
            return False

    def _load_cti_files(self) -> None:
        """搜尋並載入 .cti GenTL producer 檔案."""
        if self._harvester is None:
            return

        loaded_count = 0

        # 從環境變數讀取路徑
        env_paths = os.environ.get("GENICAM_GENTL64_PATH", "")
        search_dirs: List[str] = []
        if env_paths:
            search_dirs.extend(env_paths.split(os.pathsep))
        search_dirs.extend(_CTI_SEARCH_PATHS)

        for dir_path in search_dirs:
            p = Path(dir_path)
            if not p.is_dir():
                continue
            for cti_file in p.glob("*.cti"):
                try:
                    self._harvester.add_file(str(cti_file))
                    loaded_count += 1
                    logger.debug("載入 CTI producer: %s", cti_file)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("無法載入 CTI: %s (%s)", cti_file, exc)

        if loaded_count == 0:
            logger.info("未找到任何 .cti GenTL producer 檔案")

    def _detect_interface_type(self, dev_info) -> str:
        """根據 device_info 判斷介面類型 (GigE / USB3 / GenTL)."""
        try:
            tl_type = getattr(dev_info, "tl_type", "")
            if tl_type:
                tl_lower = tl_type.lower()
                if "gev" in tl_lower or "gige" in tl_lower:
                    return "GigE"
                if "u3v" in tl_lower or "usb3" in tl_lower:
                    return "USB3"
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to detect interface type: %s", exc)
        return "GenTL"

    # ----- 公開介面 -----

    def discover(self) -> List[CameraInfo]:
        """探索所有 GenTL 可存取的相機."""
        if not self._ensure_harvester():
            return []

        cameras: List[CameraInfo] = []
        for idx, dev_info in enumerate(self._harvester.device_info_list):
            vendor = getattr(dev_info, "vendor", "Unknown")
            model = getattr(dev_info, "model", f"Device_{idx}")
            serial = getattr(dev_info, "serial_number", "")
            display_name = getattr(dev_info, "display_name", f"{vendor} {model}")
            iface = self._detect_interface_type(dev_info)

            cam_id = f"gentl:{serial}" if serial else f"gentl:{idx}"
            cameras.append(
                CameraInfo(
                    id=cam_id,
                    name=display_name,
                    vendor=vendor,
                    model=model,
                    serial=serial or str(idx),
                    interface=iface,
                    is_available=True,
                )
            )

        logger.info("GenTL 探索到 %d 台相機", len(cameras))
        return cameras

    def open(self, camera_id: str) -> None:
        """開啟指定的 GenTL 相機."""
        with self._lock:
            if self._acquirer is not None:
                self.close()

            if not self._ensure_harvester():
                raise RuntimeError("GenTL Harvester 不可用")

            # 解析 camera_id -- 格式: "gentl:<serial>" 或 "gentl:<index>"
            key = camera_id.replace("gentl:", "", 1) if camera_id.startswith("gentl:") else camera_id

            try:
                idx = int(key)
                self._acquirer = self._harvester.create(idx)
            except ValueError:
                # 以序號搜尋
                found = False
                for i, dev in enumerate(self._harvester.device_info_list):
                    serial = getattr(dev, "serial_number", "")
                    if serial == key:
                        self._acquirer = self._harvester.create(i)
                        found = True
                        break
                if not found:
                    raise ValueError(f"找不到 GenTL 相機: {camera_id}")

            self._acquirer.start()
            self._frame_count = 0

            # 記錄已開啟相機資訊
            dev_info = self._acquirer.device.node_map
            self._current_camera = CameraInfo(
                id=camera_id,
                name=camera_id,
                vendor=self._safe_node_read(dev_info, "DeviceVendorName", "Unknown"),
                model=self._safe_node_read(dev_info, "DeviceModelName", "Unknown"),
                serial=self._safe_node_read(dev_info, "DeviceSerialNumber", ""),
                interface="GenTL",
            )
            logger.info("已開啟 GenTL 相機: %s", camera_id)

    @staticmethod
    def _safe_node_read(node_map, name: str, default: str = "") -> str:
        """安全讀取 GenICam 節點值."""
        try:
            node = node_map.get_node(name)
            if node is not None:
                return str(node.value)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to read GenICam node '%s': %s", name, exc)
        return default

    def close(self) -> None:
        """關閉相機並釋放資源."""
        with self._lock:
            if self._streaming:
                self.stop_streaming()
            if self._acquirer is not None:
                try:
                    self._acquirer.stop()
                    self._acquirer.destroy()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("GenTL 相機關閉異常: %s", exc)
                finally:
                    self._acquirer = None
                    self._current_camera = None
                    self._frame_count = 0
            logger.info("GenTL 相機已關閉")

    def grab_single(self) -> Optional[FrameResult]:
        """擷取單張影格 (帶逾時)."""
        if self._acquirer is None:
            logger.error("相機尚未開啟, 無法擷取影格")
            return None

        try:
            with self._acquirer.fetch(timeout=_GRAB_TIMEOUT_S) as buffer:
                payload = buffer.payload
                component = payload.components[0]

                raw = component.data.copy()
                pixel_format = str(getattr(component, "pixel_format", "Mono8"))
                w = component.width
                h = component.height

                image = _convert_pixel_format(raw, pixel_format, w, h)
                self._frame_count += 1

                return FrameResult(
                    image=image,
                    timestamp=time.time(),
                    frame_id=self._frame_count,
                    width=w,
                    height=h,
                    exposure_us=self._read_exposure(),
                    gain_db=self._read_gain(),
                )
        except Exception as exc:  # noqa: BLE001
            logger.error("GenTL 擷取影格失敗: %s", exc)
            return None

    def start_streaming(self, callback: Callable[[FrameResult], None]) -> None:
        """啟動背景串流執行緒."""
        if self._streaming:
            logger.warning("GenTL 串流已在執行中")
            return
        if self._acquirer is None:
            raise RuntimeError("相機尚未開啟, 無法啟動串流")

        self._stop_event.clear()
        self._streaming = True
        self._stream_thread = threading.Thread(
            target=self._stream_loop,
            args=(callback,),
            daemon=True,
            name="GenTLStream",
        )
        self._stream_thread.start()
        logger.info("GenTL 串流已啟動")

    def _stream_loop(self, callback: Callable[[FrameResult], None]) -> None:
        """串流迴圈 -- 在背景執行緒中持續擷取影格."""
        while not self._stop_event.is_set():
            frame = self.grab_single()
            if frame is not None:
                try:
                    callback(frame)
                except Exception as exc:  # noqa: BLE001
                    logger.error("串流 callback 執行異常: %s", exc)
            else:
                time.sleep(_STREAM_POLL_INTERVAL)
        self._streaming = False

    def stop_streaming(self) -> None:
        """停止串流."""
        self._stop_event.set()
        if self._stream_thread is not None and self._stream_thread.is_alive():
            self._stream_thread.join(timeout=3.0)
            if self._stream_thread.is_alive():
                logger.warning("GenTL 串流執行緒未能在時限內停止")
        self._streaming = False
        self._stream_thread = None
        logger.info("GenTL 串流已停止")

    def set_exposure(self, us: float) -> None:
        """設定曝光時間 (微秒)."""
        if self._acquirer is None:
            raise RuntimeError("相機尚未開啟")
        try:
            nm = self._acquirer.device.node_map
            nm.get_node("ExposureTime").value = us
            logger.debug("GenTL 曝光設定為 %.1f us", us)
        except Exception as exc:  # noqa: BLE001
            logger.error("設定曝光失敗: %s", exc)
            raise

    def set_gain(self, db: float) -> None:
        """設定增益 (dB)."""
        if self._acquirer is None:
            raise RuntimeError("相機尚未開啟")
        try:
            nm = self._acquirer.device.node_map
            nm.get_node("Gain").value = db
            logger.debug("GenTL 增益設定為 %.1f dB", db)
        except Exception as exc:  # noqa: BLE001
            logger.error("設定增益失敗: %s", exc)
            raise

    def get_exposure_range(self) -> Tuple[float, float]:
        """取得曝光時間範圍 (min_us, max_us)."""
        if self._acquirer is None:
            return (0.0, 0.0)
        try:
            node = self._acquirer.device.node_map.get_node("ExposureTime")
            return (float(node.min), float(node.max))
        except Exception:  # noqa: BLE001
            return (0.0, 0.0)

    def get_gain_range(self) -> Tuple[float, float]:
        """取得增益範圍 (min_db, max_db)."""
        if self._acquirer is None:
            return (0.0, 0.0)
        try:
            node = self._acquirer.device.node_map.get_node("Gain")
            return (float(node.min), float(node.max))
        except Exception:  # noqa: BLE001
            return (0.0, 0.0)

    def _read_exposure(self) -> float:
        """讀取目前曝光值 (微秒)."""
        try:
            node = self._acquirer.device.node_map.get_node("ExposureTime")
            return float(node.value)
        except Exception:  # noqa: BLE001
            return 0.0

    def _read_gain(self) -> float:
        """讀取目前增益值 (dB)."""
        try:
            node = self._acquirer.device.node_map.get_node("Gain")
            return float(node.value)
        except Exception:  # noqa: BLE001
            return 0.0

    @property
    def is_open(self) -> bool:
        return self._acquirer is not None

    @property
    def is_streaming(self) -> bool:
        return self._streaming


# =========================================================================
# OpenCV 後端 (Webcam / RTSP fallback)
# =========================================================================

class OpenCVBackend(CameraBackend):
    """Fallback backend using OpenCV VideoCapture.

    Works with USB webcams and RTSP streams. Always available as long as
    ``cv2`` is installed.
    """

    def __init__(self) -> None:
        self._cap = None  # cv2.VideoCapture
        self._streaming = False
        self._stream_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._frame_count = 0
        self._camera_id: Optional[str] = None
        self._lock = threading.Lock()
        self._current_camera: Optional[CameraInfo] = None

    # ----- 公開介面 -----

    def discover(self) -> List[CameraInfo]:
        """探測 index 0 ~ (_PROBE_MAX_INDEX - 1) 的可用 webcam."""
        import cv2

        cameras: List[CameraInfo] = []
        for idx in range(_PROBE_MAX_INDEX):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cameras.append(
                    CameraInfo(
                        id=str(idx),
                        name=f"Webcam #{idx} ({w}x{h})",
                        vendor="Generic",
                        model=f"USB Camera {idx}",
                        serial=str(idx),
                        interface="Webcam",
                        is_available=True,
                    )
                )
                cap.release()
            else:
                cap.release()

        logger.info("OpenCV 探索到 %d 台 webcam", len(cameras))
        return cameras

    def open(self, camera_id: str) -> None:
        """開啟 webcam (index) 或 RTSP 串流 (URL)."""
        import cv2

        with self._lock:
            if self._cap is not None:
                self.close()

            # 判斷是 index 還是 URL
            try:
                source = int(camera_id)
            except ValueError:
                source = camera_id  # RTSP / HTTP URL

            self._cap = cv2.VideoCapture(source)
            if not self._cap.isOpened():
                self._cap.release()
                self._cap = None
                raise RuntimeError(f"無法開啟 OpenCV 相機: {camera_id}")

            self._camera_id = camera_id
            self._frame_count = 0

            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            is_url = isinstance(source, str)
            self._current_camera = CameraInfo(
                id=camera_id,
                name=f"RTSP Stream" if is_url else f"Webcam #{source}",
                vendor="Generic",
                model="Stream" if is_url else f"USB Camera {source}",
                serial=camera_id,
                interface="RTSP" if is_url else "Webcam",
            )
            logger.info("已開啟 OpenCV 相機: %s (%dx%d)", camera_id, w, h)

    def close(self) -> None:
        """關閉 VideoCapture."""
        with self._lock:
            if self._streaming:
                self.stop_streaming()
            if self._cap is not None:
                self._cap.release()
                self._cap = None
                self._camera_id = None
                self._current_camera = None
                self._frame_count = 0
            logger.info("OpenCV 相機已關閉")

    def grab_single(self) -> Optional[FrameResult]:
        """擷取單張影格."""
        if self._cap is None or not self._cap.isOpened():
            logger.error("OpenCV 相機尚未開啟")
            return None

        ret, frame = self._cap.read()
        if not ret or frame is None:
            logger.warning("OpenCV 讀取影格失敗")
            return None

        self._frame_count += 1
        h, w = frame.shape[:2]

        return FrameResult(
            image=frame,
            timestamp=time.time(),
            frame_id=self._frame_count,
            width=w,
            height=h,
            exposure_us=self._read_exposure(),
            gain_db=self._read_gain(),
        )

    def start_streaming(self, callback: Callable[[FrameResult], None]) -> None:
        """啟動背景串流."""
        if self._streaming:
            logger.warning("OpenCV 串流已在執行中")
            return
        if self._cap is None:
            raise RuntimeError("相機尚未開啟, 無法啟動串流")

        self._stop_event.clear()
        self._streaming = True
        self._stream_thread = threading.Thread(
            target=self._stream_loop,
            args=(callback,),
            daemon=True,
            name="OpenCVStream",
        )
        self._stream_thread.start()
        logger.info("OpenCV 串流已啟動")

    def _stream_loop(self, callback: Callable[[FrameResult], None]) -> None:
        """串流迴圈."""
        while not self._stop_event.is_set():
            frame = self.grab_single()
            if frame is not None:
                try:
                    callback(frame)
                except Exception as exc:  # noqa: BLE001
                    logger.error("串流 callback 執行異常: %s", exc)
            else:
                # 讀取失敗時短暫等待避免 busy loop
                time.sleep(_STREAM_POLL_INTERVAL)

    def stop_streaming(self) -> None:
        """停止串流."""
        self._stop_event.set()
        if self._stream_thread is not None and self._stream_thread.is_alive():
            self._stream_thread.join(timeout=3.0)
            if self._stream_thread.is_alive():
                logger.warning("OpenCV 串流執行緒未能在時限內停止")
        self._streaming = False
        self._stream_thread = None
        logger.info("OpenCV 串流已停止")

    def set_exposure(self, us: float) -> None:
        """設定曝光時間 (微秒) -- 透過 OpenCV CAP_PROP_EXPOSURE."""
        import cv2

        if self._cap is None:
            raise RuntimeError("相機尚未開啟")
        # OpenCV exposure 單位因驅動而異, 這裡嘗試以微秒設定
        self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 手動模式
        self._cap.set(cv2.CAP_PROP_EXPOSURE, us)
        logger.debug("OpenCV 曝光設定為 %.1f (驅動單位)", us)

    def set_gain(self, db: float) -> None:
        """設定增益 (dB) -- 透過 OpenCV CAP_PROP_GAIN."""
        import cv2

        if self._cap is None:
            raise RuntimeError("相機尚未開啟")
        self._cap.set(cv2.CAP_PROP_GAIN, db)
        logger.debug("OpenCV 增益設定為 %.1f", db)

    def get_exposure_range(self) -> Tuple[float, float]:
        """取得曝光範圍 -- OpenCV 無標準方式, 回傳預設值."""
        return (1.0, 1_000_000.0)

    def get_gain_range(self) -> Tuple[float, float]:
        """取得增益範圍 -- OpenCV 無標準方式, 回傳預設值."""
        return (0.0, 48.0)

    def _read_exposure(self) -> float:
        """讀取目前曝光值."""
        import cv2

        if self._cap is None:
            return 0.0
        try:
            return float(self._cap.get(cv2.CAP_PROP_EXPOSURE))
        except Exception:  # noqa: BLE001
            return 0.0

    def _read_gain(self) -> float:
        """讀取目前增益值."""
        import cv2

        if self._cap is None:
            return 0.0
        try:
            return float(self._cap.get(cv2.CAP_PROP_GAIN))
        except Exception:  # noqa: BLE001
            return 0.0

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    @property
    def is_streaming(self) -> bool:
        return self._streaming


# =========================================================================
# 統一相機管理器
# =========================================================================

class CameraManager:
    """Unified camera manager that auto-discovers backends.

    Usage::

        mgr = CameraManager()
        cameras = mgr.discover_all()
        mgr.open(cameras[0].id)
        frame = mgr.grab_single()
        mgr.start_streaming(on_frame_callback)
        mgr.stop_streaming()
        mgr.close()
    """

    def __init__(self) -> None:
        self._backends: Dict[str, CameraBackend] = {}
        self._active_backend: Optional[CameraBackend] = None
        self._active_camera: Optional[CameraInfo] = None
        self._camera_registry: Dict[str, Tuple[CameraBackend, CameraInfo]] = {}
        self._lock = threading.Lock()
        self._trigger_mode: str = "freerun"

        # 嘗試建立 GenTL 後端
        try:
            gentl = GenTLBackend()
            self._backends["gentl"] = gentl
            logger.debug("GenTL 後端已註冊")
        except Exception as exc:  # noqa: BLE001
            logger.debug("GenTL 後端初始化失敗 (非必要): %s", exc)

        # OpenCV 後端 -- 永遠可用
        self._backends["opencv"] = OpenCVBackend()
        logger.debug("OpenCV 後端已註冊")

    def _ensure_active(self) -> CameraBackend:
        """確認有作用中的後端, 否則拋出例外."""
        if self._active_backend is None:
            raise RuntimeError("尚未開啟任何相機, 請先呼叫 open()")
        return self._active_backend

    # ----- 探索 -----

    @log_operation(logger)
    def discover_all(self) -> List[CameraInfo]:
        """探索所有後端的可用相機."""
        all_cameras: List[CameraInfo] = []
        self._camera_registry.clear()

        for name, backend in self._backends.items():
            try:
                cameras = backend.discover()
                for cam in cameras:
                    self._camera_registry[cam.id] = (backend, cam)
                all_cameras.extend(cameras)
            except Exception as exc:  # noqa: BLE001
                logger.warning("後端 '%s' 探索失敗: %s", name, exc)

        logger.info("共探索到 %d 台相機", len(all_cameras))
        return all_cameras

    # ----- 開啟 / 關閉 -----

    @log_operation(logger)
    def open(self, camera_id: str) -> None:
        """開啟指定相機 (自動選擇對應後端)."""
        with self._lock:
            if self._active_backend is not None:
                self.close()

            # 查詢已註冊的相機
            if camera_id in self._camera_registry:
                backend, cam_info = self._camera_registry[camera_id]
                backend.open(camera_id)
                self._active_backend = backend
                self._active_camera = cam_info
                logger.info("已開啟相機: %s (後端: %s)", cam_info, type(backend).__name__)
                return

            # 未在註冊表中 -- 根據 ID 格式推斷後端
            if camera_id.startswith("gentl:"):
                backend = self._backends.get("gentl")
                if backend is None:
                    raise RuntimeError("GenTL 後端不可用")
                backend.open(camera_id)
                self._active_backend = backend
            else:
                # 預設使用 OpenCV
                backend = self._backends["opencv"]
                backend.open(camera_id)
                self._active_backend = backend

            self._active_camera = CameraInfo(
                id=camera_id,
                name=camera_id,
                vendor="Unknown",
                model="Unknown",
                serial=camera_id,
                interface="Unknown",
            )
            logger.info("已開啟相機: %s", camera_id)

    @log_operation(logger)
    def close(self) -> None:
        """關閉目前開啟的相機."""
        with self._lock:
            if self._active_backend is not None:
                self._active_backend.close()
                self._active_backend = None
                self._active_camera = None

    # ----- 擷取 -----

    @log_operation(logger)
    def grab_single(self) -> Optional[FrameResult]:
        """擷取單張影格."""
        backend = self._ensure_active()
        frame = backend.grab_single()
        if frame is not None:
            validate_image(frame.image, "grabbed frame")
        return frame

    # ----- 串流 -----

    def start_streaming(self, callback: Callable[[FrameResult], None]) -> None:
        """啟動連續串流."""
        backend = self._ensure_active()
        backend.start_streaming(callback)
        logger.info("串流已啟動")

    def stop_streaming(self) -> None:
        """停止串流."""
        backend = self._ensure_active()
        backend.stop_streaming()
        logger.info("串流已停止")

    # ----- 曝光 / 增益 -----

    def set_exposure(self, us: float) -> None:
        """設定曝光時間 (微秒)."""
        backend = self._ensure_active()
        backend.set_exposure(us)

    def set_gain(self, db: float) -> None:
        """設定增益 (dB)."""
        backend = self._ensure_active()
        backend.set_gain(db)

    def get_exposure_range(self) -> Tuple[float, float]:
        """取得曝光範圍."""
        backend = self._ensure_active()
        return backend.get_exposure_range()

    def get_gain_range(self) -> Tuple[float, float]:
        """取得增益範圍."""
        backend = self._ensure_active()
        return backend.get_gain_range()

    # ----- 觸發模式 -----

    def set_trigger_mode(self, mode: str) -> None:
        """設定觸發模式.

        Args:
            mode: "freerun" (連續), "software" (軟體觸發), "hardware" (硬體觸發)
        """
        valid_modes = ("freerun", "software", "hardware")
        if mode not in valid_modes:
            raise ValueError(f"無效的觸發模式: {mode!r}, 必須是 {valid_modes} 之一")

        self._trigger_mode = mode

        # 如果使用 GenTL 後端, 嘗試寫入 GenICam 節點
        if isinstance(self._active_backend, GenTLBackend) and self._active_backend.is_open:
            try:
                acq = self._active_backend._acquirer
                nm = acq.device.node_map
                if mode == "freerun":
                    nm.get_node("TriggerMode").value = "Off"
                elif mode == "software":
                    nm.get_node("TriggerMode").value = "On"
                    nm.get_node("TriggerSource").value = "Software"
                elif mode == "hardware":
                    nm.get_node("TriggerMode").value = "On"
                    nm.get_node("TriggerSource").value = "Line1"
            except Exception as exc:  # noqa: BLE001
                logger.warning("設定觸發模式失敗 (GenICam): %s", exc)

        logger.info("觸發模式已設定為: %s", mode)

    def software_trigger(self) -> Optional[FrameResult]:
        """送出軟體觸發並等待影格.

        Returns:
            FrameResult 或 None (若觸發失敗)
        """
        if self._trigger_mode != "software":
            logger.warning("目前非軟體觸發模式 (mode=%s), 仍嘗試擷取", self._trigger_mode)

        # GenTL: 送出 TriggerSoftware 命令
        if isinstance(self._active_backend, GenTLBackend) and self._active_backend.is_open:
            try:
                nm = self._active_backend._acquirer.device.node_map
                nm.get_node("TriggerSoftware").execute()
            except Exception as exc:  # noqa: BLE001
                logger.error("軟體觸發命令失敗: %s", exc)
                return None

        return self.grab_single()

    # ----- 狀態屬性 -----

    @property
    def is_open(self) -> bool:
        """相機是否已開啟."""
        return self._active_backend is not None and self._active_backend.is_open

    @property
    def is_streaming(self) -> bool:
        """是否正在串流."""
        return self._active_backend is not None and self._active_backend.is_streaming

    @property
    def current_camera(self) -> Optional[CameraInfo]:
        """目前開啟的相機資訊."""
        return self._active_camera

    def __enter__(self) -> "CameraManager":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __repr__(self) -> str:
        status = "open" if self.is_open else "closed"
        cam = self._active_camera
        cam_str = str(cam) if cam else "None"
        return f"CameraManager(status={status}, camera={cam_str})"


# =========================================================================
# 公開便利函式
# =========================================================================

@log_operation(logger)
def list_cameras() -> List[CameraInfo]:
    """Discover all available cameras across all backends.

    Returns:
        所有可用相機的列表
    """
    mgr = CameraManager()
    return mgr.discover_all()


@log_operation(logger)
def grab_image(camera_id: str = "0") -> Optional[np.ndarray]:
    """Quick single-shot grab from a camera.

    Args:
        camera_id: 相機 ID (預設 "0" 為第一個 webcam)

    Returns:
        BGR 影像 numpy 陣列, 或 None (若擷取失敗)
    """
    mgr = CameraManager()
    try:
        mgr.open(camera_id)
        frame = mgr.grab_single()
        return frame.image if frame is not None else None
    finally:
        mgr.close()


def create_camera_manager() -> CameraManager:
    """Factory function to create a CameraManager instance.

    Returns:
        新的 CameraManager 實例
    """
    return CameraManager()


# =========================================================================
# 模組自檢 (直接執行時列出可用相機)
# =========================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("=== Camera Discovery ===")
    cameras = list_cameras()
    if not cameras:
        print("  (未偵測到任何相機)")
    for cam in cameras:
        print(f"  {cam}")

    # 嘗試從第一台 webcam 擷取一張影像
    if cameras:
        first = cameras[0]
        print(f"\n=== Grab from {first.id} ===")
        img = grab_image(first.id)
        if img is not None:
            print(f"  成功擷取影像: shape={img.shape}, dtype={img.dtype}")
        else:
            print("  擷取失敗")
