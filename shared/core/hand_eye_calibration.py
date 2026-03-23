"""
core/hand_eye_calibration.py - Hand-eye calibration and multi-camera extrinsics.

Wraps OpenCV hand-eye calibration APIs (Tsai-Lenz, Park-Martin, etc.) with a
Vision-style interface for robotic vision pipelines.  Supports both eye-in-hand
(camera mounted on robot end-effector) and eye-to-hand (camera fixed in the
workspace) configurations, as well as stereo multi-camera extrinsic calibration
and rigid-body transform chain computation.

Categories:
    1. Data Classes (HandEyeResult, PoseStamped, StereoPairResult)
    2. Pose Representation Utilities
    3. Eye-in-Hand Calibration
    4. Eye-to-Hand Calibration
    5. Multi-Camera Extrinsic Calibration
    6. Transform Chain Computation
    7. JSON Persistence
    8. Validation Helpers
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ====================================================================== #
#  Named constants                                                        #
# ====================================================================== #

# OpenCV hand-eye method mapping
HANDEYE_METHODS: Dict[str, int] = {
    "tsai": cv2.CALIB_HAND_EYE_TSAI,
    "park": cv2.CALIB_HAND_EYE_PARK,
    "horaud": cv2.CALIB_HAND_EYE_HORAUD,
    "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
    "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
}

DEFAULT_METHOD = "tsai"
REPROJECTION_WARN_THRESHOLD = 1.0  # pixels


# ====================================================================== #
#  Internal helpers                                                       #
# ====================================================================== #


def _ensure_rotation_matrix(R: np.ndarray) -> np.ndarray:
    """Validate and return a 3x3 rotation matrix."""
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3):
        raise ValueError(f"Expected 3x3 rotation matrix, got shape {R.shape}")
    return R


def _ensure_translation(t: np.ndarray) -> np.ndarray:
    """Validate and return a (3,1) translation vector."""
    t = np.asarray(t, dtype=np.float64).reshape(3, 1)
    return t


def _make_4x4(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Compose a 4x4 homogeneous transformation matrix from R and t."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t.ravel()
    return T


def _decompose_4x4(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Extract R (3x3) and t (3x1) from a 4x4 homogeneous matrix."""
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"Expected 4x4 matrix, got shape {T.shape}")
    return T[:3, :3].copy(), T[:3, 3].reshape(3, 1).copy()


# ====================================================================== #
#  Pose Representation Utilities                                          #
# ====================================================================== #


def rotation_matrix_to_euler(R: np.ndarray) -> Tuple[float, float, float]:
    """Convert a 3x3 rotation matrix to Euler angles (roll, pitch, yaw).

    Uses the ZYX (Tait-Bryan) convention.  Angles are returned in radians.
    """
    R = _ensure_rotation_matrix(R)
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0.0
    return (roll, pitch, yaw)


def euler_to_rotation_matrix(
    roll: float, pitch: float, yaw: float
) -> np.ndarray:
    """Convert Euler angles (ZYX Tait-Bryan, radians) to a 3x3 rotation matrix."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    R = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )
    return R


def rotation_matrix_to_quaternion(R: np.ndarray) -> Tuple[float, float, float, float]:
    """Convert a 3x3 rotation matrix to a unit quaternion (w, x, y, z).

    Uses Shepperd's method for numerical stability.
    """
    R = _ensure_rotation_matrix(R)
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    # Normalise
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    return (w / norm, x / norm, y / norm, z / norm)


def quaternion_to_rotation_matrix(
    w: float, x: float, y: float, z: float
) -> np.ndarray:
    """Convert a unit quaternion (w, x, y, z) to a 3x3 rotation matrix."""
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm

    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return R


def rodrigues_to_rotation_matrix(rvec: np.ndarray) -> np.ndarray:
    """Convert a Rodrigues rotation vector (3,) to a 3x3 rotation matrix."""
    rvec = np.asarray(rvec, dtype=np.float64).ravel()
    R, _ = cv2.Rodrigues(rvec)
    return R


def rotation_matrix_to_rodrigues(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a Rodrigues rotation vector (3,)."""
    R = _ensure_rotation_matrix(R)
    rvec, _ = cv2.Rodrigues(R)
    return rvec.ravel()


# ====================================================================== #
#  Data Classes                                                           #
# ====================================================================== #


@dataclass
class HandEyeResult:
    """Result of a hand-eye calibration.

    Attributes:
        transformation: 4x4 homogeneous transformation matrix (camera-to-gripper
            for eye-in-hand, or camera-to-base for eye-to-hand).
        rotation:       3x3 rotation sub-matrix.
        translation:    (3,) translation vector in metres.
        euler_angles:   (roll, pitch, yaw) in radians (ZYX convention).
        quaternion:     (w, x, y, z) unit quaternion.
        reprojection_error: Mean reprojection error in pixels (if computed).
        method:         Name of the solver method used.
        num_poses:      Number of pose pairs used for calibration.
    """

    transformation: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray
    euler_angles: Tuple[float, float, float]
    quaternion: Tuple[float, float, float, float]
    reprojection_error: float
    method: str
    num_poses: int


@dataclass
class StereoPairResult:
    """Result of multi-camera (stereo) extrinsic calibration.

    Attributes:
        R:              3x3 rotation from camera 1 to camera 2.
        t:              (3,1) translation from camera 1 to camera 2.
        E:              3x3 essential matrix.
        F:              3x3 fundamental matrix.
        transformation: 4x4 homogeneous transformation cam1 -> cam2.
        reprojection_error: RMS stereo reprojection error in pixels.
    """

    R: np.ndarray
    t: np.ndarray
    E: np.ndarray
    F: np.ndarray
    transformation: np.ndarray
    reprojection_error: float


# ====================================================================== #
#  Eye-in-Hand Calibration                                                #
# ====================================================================== #


def calibrate_eye_in_hand(
    R_gripper2base: Sequence[np.ndarray],
    t_gripper2base: Sequence[np.ndarray],
    R_target2cam: Sequence[np.ndarray],
    t_target2cam: Sequence[np.ndarray],
    method: str = DEFAULT_METHOD,
) -> HandEyeResult:
    """Solve AX = XB for eye-in-hand configuration.

    The camera is mounted on the robot gripper.  Given a set of paired
    gripper-to-base and target-to-camera poses, solves for the camera-to-
    gripper transformation X.

    Args:
        R_gripper2base: List of 3x3 rotation matrices (gripper -> base).
        t_gripper2base: List of (3,1) translation vectors (gripper -> base).
        R_target2cam:   List of 3x3 rotation matrices (target -> camera).
        t_target2cam:   List of (3,1) translation vectors (target -> camera).
        method:         Solver name: ``"tsai"``, ``"park"``, ``"horaud"``,
                        ``"andreff"``, or ``"daniilidis"``.

    Returns:
        HandEyeResult with the camera-to-gripper transformation.
    """
    n = len(R_gripper2base)
    if n < 3:
        raise ValueError(f"At least 3 pose pairs required, got {n}")
    if not (n == len(t_gripper2base) == len(R_target2cam) == len(t_target2cam)):
        raise ValueError("All input lists must have the same length")

    method_key = method.lower()
    if method_key not in HANDEYE_METHODS:
        raise ValueError(
            f"Unknown method '{method}'. Choose from {list(HANDEYE_METHODS)}"
        )

    # Ensure correct shapes
    R_g = [_ensure_rotation_matrix(r) for r in R_gripper2base]
    t_g = [_ensure_translation(t) for t in t_gripper2base]
    R_t = [_ensure_rotation_matrix(r) for r in R_target2cam]
    t_t = [_ensure_translation(t) for t in t_target2cam]

    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_g, t_g, R_t, t_t, method=HANDEYE_METHODS[method_key]
    )

    T = _make_4x4(R_cam2gripper, t_cam2gripper)
    euler = rotation_matrix_to_euler(R_cam2gripper)
    quat = rotation_matrix_to_quaternion(R_cam2gripper)

    reproj = _compute_handeye_reprojection_error(R_g, t_g, R_t, t_t, T)

    if reproj > REPROJECTION_WARN_THRESHOLD:
        logger.warning(
            "Hand-eye reprojection error %.4f px exceeds threshold %.1f px",
            reproj,
            REPROJECTION_WARN_THRESHOLD,
        )

    logger.info(
        "Eye-in-hand calibration (%s): reproj=%.4f px, %d poses",
        method_key,
        reproj,
        n,
    )

    return HandEyeResult(
        transformation=T,
        rotation=R_cam2gripper,
        translation=t_cam2gripper.ravel(),
        euler_angles=euler,
        quaternion=quat,
        reprojection_error=reproj,
        method=method_key,
        num_poses=n,
    )


# ====================================================================== #
#  Eye-to-Hand Calibration                                                #
# ====================================================================== #


def calibrate_eye_to_hand(
    R_gripper2base: Sequence[np.ndarray],
    t_gripper2base: Sequence[np.ndarray],
    R_target2cam: Sequence[np.ndarray],
    t_target2cam: Sequence[np.ndarray],
    method: str = DEFAULT_METHOD,
) -> HandEyeResult:
    """Solve AX = ZB for eye-to-hand configuration.

    The camera is fixed in the workspace and observes both the calibration
    target (attached to the gripper) and the robot base.  Solves for the
    camera-to-base transformation.

    Args:
        R_gripper2base: List of 3x3 rotation matrices (gripper -> base).
        t_gripper2base: List of (3,1) translation vectors (gripper -> base).
        R_target2cam:   List of 3x3 rotation matrices (target -> camera).
        t_target2cam:   List of (3,1) translation vectors (target -> camera).
        method:         Solver name (same as ``calibrate_eye_in_hand``).

    Returns:
        HandEyeResult with the camera-to-base transformation.
    """
    n = len(R_gripper2base)
    if n < 3:
        raise ValueError(f"At least 3 pose pairs required, got {n}")
    if not (n == len(t_gripper2base) == len(R_target2cam) == len(t_target2cam)):
        raise ValueError("All input lists must have the same length")

    method_key = method.lower()
    if method_key not in HANDEYE_METHODS:
        raise ValueError(
            f"Unknown method '{method}'. Choose from {list(HANDEYE_METHODS)}"
        )

    # For eye-to-hand: invert gripper-to-base to get base-to-gripper
    R_base2gripper = []
    t_base2gripper = []
    for R, t in zip(R_gripper2base, t_gripper2base):
        R = _ensure_rotation_matrix(R)
        t = _ensure_translation(t)
        R_inv = R.T
        t_inv = -R_inv @ t
        R_base2gripper.append(R_inv)
        t_base2gripper.append(t_inv)

    R_t = [_ensure_rotation_matrix(r) for r in R_target2cam]
    t_t = [_ensure_translation(t) for t in t_target2cam]

    R_cam2base, t_cam2base = cv2.calibrateHandEye(
        R_base2gripper,
        t_base2gripper,
        R_t,
        t_t,
        method=HANDEYE_METHODS[method_key],
    )

    T = _make_4x4(R_cam2base, t_cam2base)
    euler = rotation_matrix_to_euler(R_cam2base)
    quat = rotation_matrix_to_quaternion(R_cam2base)

    reproj = _compute_handeye_reprojection_error(
        R_base2gripper, t_base2gripper, R_t, t_t, T
    )

    logger.info(
        "Eye-to-hand calibration (%s): reproj=%.4f px, %d poses",
        method_key,
        reproj,
        n,
    )

    return HandEyeResult(
        transformation=T,
        rotation=R_cam2base,
        translation=t_cam2base.ravel(),
        euler_angles=euler,
        quaternion=quat,
        reprojection_error=reproj,
        method=method_key,
        num_poses=n,
    )


# ====================================================================== #
#  Multi-Camera Extrinsic Calibration                                     #
# ====================================================================== #


def calibrate_stereo_extrinsics(
    object_points: Sequence[np.ndarray],
    image_points_1: Sequence[np.ndarray],
    image_points_2: Sequence[np.ndarray],
    camera_matrix_1: np.ndarray,
    dist_coeffs_1: np.ndarray,
    camera_matrix_2: np.ndarray,
    dist_coeffs_2: np.ndarray,
    image_size: Tuple[int, int],
    flags: int = cv2.CALIB_FIX_INTRINSIC,
) -> StereoPairResult:
    """Calibrate extrinsic parameters of a stereo camera pair.

    Args:
        object_points:  List of Nx3 arrays of 3-D calibration pattern points.
        image_points_1: List of Nx2 arrays of 2-D points in camera 1.
        image_points_2: List of Nx2 arrays of 2-D points in camera 2.
        camera_matrix_1: 3x3 intrinsic matrix of camera 1.
        dist_coeffs_1:  Distortion coefficients of camera 1.
        camera_matrix_2: 3x3 intrinsic matrix of camera 2.
        dist_coeffs_2:  Distortion coefficients of camera 2.
        image_size:     (width, height) of the images.
        flags:          OpenCV stereo calibration flags.

    Returns:
        StereoPairResult with rotation, translation and fundamental/essential
        matrices for the camera pair.
    """
    if len(object_points) < 3:
        raise ValueError("At least 3 image pairs required for stereo calibration")

    rms, K1, d1, K2, d2, R, t, E, F = cv2.stereoCalibrate(
        object_points,
        image_points_1,
        image_points_2,
        camera_matrix_1,
        dist_coeffs_1,
        camera_matrix_2,
        dist_coeffs_2,
        image_size,
        flags=flags,
    )

    T = _make_4x4(R, t)
    logger.info("Stereo calibration: RMS=%.4f px", rms)

    return StereoPairResult(
        R=R,
        t=t,
        E=E,
        F=F,
        transformation=T,
        reprojection_error=rms,
    )


# ====================================================================== #
#  Transform Chain Computation                                            #
# ====================================================================== #


def chain_transforms(*transforms: np.ndarray) -> np.ndarray:
    """Compute the composition of a sequence of 4x4 transformations.

    ``chain_transforms(T_AB, T_BC, T_CD)`` returns ``T_AD = T_AB @ T_BC @ T_CD``.
    """
    if len(transforms) < 1:
        raise ValueError("At least one transformation required")

    result = np.eye(4, dtype=np.float64)
    for T in transforms:
        T = np.asarray(T, dtype=np.float64)
        if T.shape != (4, 4):
            raise ValueError(f"Expected 4x4 matrix, got shape {T.shape}")
        result = result @ T
    return result


def invert_transform(T: np.ndarray) -> np.ndarray:
    """Invert a 4x4 homogeneous transformation matrix efficiently."""
    R, t = _decompose_4x4(T)
    R_inv = R.T
    t_inv = -R_inv @ t
    return _make_4x4(R_inv, t_inv)


# ====================================================================== #
#  JSON Persistence                                                       #
# ====================================================================== #


def save_hand_eye_result(result: HandEyeResult, path: Union[str, Path]) -> None:
    """Persist a HandEyeResult to a JSON file."""
    data = {
        "transformation": result.transformation.tolist(),
        "rotation": result.rotation.tolist(),
        "translation": result.translation.tolist(),
        "euler_angles_rad": list(result.euler_angles),
        "quaternion_wxyz": list(result.quaternion),
        "reprojection_error": result.reprojection_error,
        "method": result.method,
        "num_poses": result.num_poses,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fp:
        json.dump(data, fp, indent=2)
    logger.info("Saved hand-eye result to %s", path)


def load_hand_eye_result(path: Union[str, Path]) -> HandEyeResult:
    """Load a HandEyeResult from a previously saved JSON file."""
    path = Path(path)
    with open(path) as fp:
        data = json.load(fp)

    T = np.array(data["transformation"], dtype=np.float64)
    R = np.array(data["rotation"], dtype=np.float64)
    t = np.array(data["translation"], dtype=np.float64)

    return HandEyeResult(
        transformation=T,
        rotation=R,
        translation=t,
        euler_angles=tuple(data["euler_angles_rad"]),
        quaternion=tuple(data["quaternion_wxyz"]),
        reprojection_error=data["reprojection_error"],
        method=data["method"],
        num_poses=data["num_poses"],
    )


def save_stereo_result(result: StereoPairResult, path: Union[str, Path]) -> None:
    """Persist a StereoPairResult to a JSON file."""
    data = {
        "R": result.R.tolist(),
        "t": result.t.tolist(),
        "E": result.E.tolist(),
        "F": result.F.tolist(),
        "transformation": result.transformation.tolist(),
        "reprojection_error": result.reprojection_error,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fp:
        json.dump(data, fp, indent=2)
    logger.info("Saved stereo result to %s", path)


def load_stereo_result(path: Union[str, Path]) -> StereoPairResult:
    """Load a StereoPairResult from a previously saved JSON file."""
    path = Path(path)
    with open(path) as fp:
        data = json.load(fp)

    return StereoPairResult(
        R=np.array(data["R"], dtype=np.float64),
        t=np.array(data["t"], dtype=np.float64),
        E=np.array(data["E"], dtype=np.float64),
        F=np.array(data["F"], dtype=np.float64),
        transformation=np.array(data["transformation"], dtype=np.float64),
        reprojection_error=data["reprojection_error"],
    )


# ====================================================================== #
#  Validation Helpers                                                     #
# ====================================================================== #


def _compute_handeye_reprojection_error(
    R_a: List[np.ndarray],
    t_a: List[np.ndarray],
    R_b: List[np.ndarray],
    t_b: List[np.ndarray],
    T_x: np.ndarray,
) -> float:
    """Estimate consistency error across all pose pairs.

    For each pair (i, j) with i < j, computes the relative poses and checks
    the AX = XB residual in translation space.  Returns the RMS residual
    over all pairs in the same units as the translation vectors.
    """
    n = len(R_a)
    if n < 2:
        return 0.0

    R_x, t_x = _decompose_4x4(T_x)
    errors: List[float] = []

    for i in range(n):
        T_a_i = _make_4x4(R_a[i], t_a[i])
        T_b_i = _make_4x4(R_b[i], t_b[i])
        for j in range(i + 1, n):
            T_a_j = _make_4x4(R_a[j], t_a[j])
            T_b_j = _make_4x4(R_b[j], t_b[j])

            # Relative motion
            A_ij = np.linalg.inv(T_a_i) @ T_a_j
            B_ij = T_b_i @ np.linalg.inv(T_b_j)

            # AX = XB residual
            lhs = A_ij @ T_x
            rhs = T_x @ B_ij
            residual = np.linalg.norm(lhs[:3, 3] - rhs[:3, 3])
            errors.append(residual)

    return float(np.sqrt(np.mean(np.square(errors)))) if errors else 0.0


def validate_hand_eye_result(
    result: HandEyeResult,
    max_reproj_error: float = REPROJECTION_WARN_THRESHOLD,
) -> bool:
    """Check if a hand-eye calibration result is within acceptable tolerances.

    Validates:
        1. Rotation matrix is proper (orthonormal with det = +1).
        2. Reprojection error is below the given threshold.
        3. Translation magnitude is finite and non-zero.
    """
    R = result.rotation
    # Orthonormality
    orth_err = np.linalg.norm(R @ R.T - np.eye(3))
    if orth_err > 1e-4:
        logger.warning("Rotation matrix is not orthonormal (err=%.6f)", orth_err)
        return False

    # Determinant
    det = np.linalg.det(R)
    if abs(det - 1.0) > 1e-4:
        logger.warning("Rotation matrix determinant = %.6f (expected 1.0)", det)
        return False

    # Reprojection
    if result.reprojection_error > max_reproj_error:
        logger.warning(
            "Reprojection error %.4f exceeds threshold %.4f",
            result.reprojection_error,
            max_reproj_error,
        )
        return False

    # Translation sanity
    t_norm = np.linalg.norm(result.translation)
    if not np.isfinite(t_norm) or t_norm < 1e-12:
        logger.warning("Translation vector has invalid magnitude: %.6e", t_norm)
        return False

    return True
