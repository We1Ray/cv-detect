"""
core/vision_3d.py - 3D Vision and Point Cloud Processing.

Provides self-contained 3D vision operators for depth-map and point-cloud
processing in industrial inspection pipelines.  All algorithms are
implemented using only NumPy (with optional SciPy / OpenCV acceleration).

Categories:
    1. Data Classes (PointCloud)
    2. Depth Map -> Point Cloud Conversion
    3. Surface Normal Estimation
    4. Plane Fitting (RANSAC)
    5. Height Map Extraction
    6. Point Cloud Filtering (Statistical Outlier, Radius)
    7. Voxel Grid Down-sampling
    8. 3D Distance Computation
    9. Depth Map Visualisation
   10. PLY I/O (Save / Load)
   11. Surface Roughness & Flatness Measurement
"""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from scipy.spatial import cKDTree

from shared.op_logger import log_operation
from shared.validation import validate_image, validate_positive

logger = logging.getLogger(__name__)

# ====================================================================== #
#  Named constants                                                        #
# ====================================================================== #

_DEFAULT_VOXEL_SIZE: float = 1.0
_DEFAULT_RANSAC_ITERATIONS: int = 1000
_DEFAULT_RANSAC_THRESHOLD: float = 0.01
_DEFAULT_STATISTICAL_K: int = 20
_DEFAULT_STATISTICAL_STD: float = 2.0
_DEFAULT_RADIUS: float = 0.05
_DEFAULT_MIN_NEIGHBOURS: int = 5
_PLY_HEADER_TEMPLATE = (
    "ply\n"
    "format binary_little_endian 1.0\n"
    "element vertex {n}\n"
    "property float x\n"
    "property float y\n"
    "property float z\n"
    "{extra}"
    "end_header\n"
)


# ====================================================================== #
#  Data classes                                                           #
# ====================================================================== #


@dataclass
class PointCloud:
    """Container for a 3D point cloud with optional colours and normals.

    Attributes:
        xyz:     (N, 3) float64 array of point positions.
        colors:  Optional (N, 3) uint8 BGR colour per point.
        normals: Optional (N, 3) float64 unit normal per point.
    """

    xyz: np.ndarray
    colors: Optional[np.ndarray] = None
    normals: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.xyz.ndim != 2 or self.xyz.shape[1] != 3:
            raise ValueError("xyz must be (N, 3)")
        if self.colors is not None:
            if self.colors.shape != (len(self.xyz), 3):
                raise ValueError("colors must be (N, 3) matching xyz")
        if self.normals is not None:
            if self.normals.shape != (len(self.xyz), 3):
                raise ValueError("normals must be (N, 3) matching xyz")

    def __len__(self) -> int:
        return self.xyz.shape[0]

    @property
    def centroid(self) -> np.ndarray:
        """Return the mean (x, y, z) position."""
        return self.xyz.mean(axis=0)

    def bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(min_corner, max_corner)`` of the axis-aligned bbox."""
        return self.xyz.min(axis=0), self.xyz.max(axis=0)


@dataclass
class PlaneModel:
    """Fitted plane ``ax + by + cz + d = 0``.

    Attributes:
        normal:   Unit normal (a, b, c).
        d:        Signed offset.
        inliers:  Boolean mask of inlier points used for the fit.
    """

    normal: np.ndarray
    d: float
    inliers: np.ndarray


@dataclass
class FlatnessResult:
    """Result of a flatness / roughness measurement.

    Attributes:
        rms_roughness:  Root-mean-square deviation from the best-fit plane.
        peak_to_valley: Maximum range of signed deviations.
        mean_deviation: Mean absolute deviation from the plane.
        deviations:     Per-point signed distances to the best-fit plane.
    """

    rms_roughness: float
    peak_to_valley: float
    mean_deviation: float
    deviations: np.ndarray


# ====================================================================== #
#  Internal helpers                                                       #
# ====================================================================== #


def _ensure_float(arr: np.ndarray) -> np.ndarray:
    """Cast to float64 if not already floating-point."""
    if not np.issubdtype(arr.dtype, np.floating):
        return arr.astype(np.float64)
    return arr


# ====================================================================== #
#  Depth map -> point cloud                                               #
# ====================================================================== #


@log_operation(logger)
def depth_to_point_cloud(
    depth: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    depth_scale: float = 1.0,
    color_image: Optional[np.ndarray] = None,
) -> PointCloud:
    """Convert a depth map to a 3D point cloud using camera intrinsics.

    Parameters:
        depth:        (H, W) depth image (uint16 or float).
        fx, fy:       Focal lengths in pixels.
        cx, cy:       Principal point in pixels.
        depth_scale:  Multiplicative factor to convert depth pixels to metres.
        color_image:  Optional (H, W, 3) BGR image for colouring.

    Returns:
        A :class:`PointCloud` instance with valid (non-zero depth) points.
    """
    validate_image(depth, "depth")
    validate_positive(fx, "fx")
    validate_positive(fy, "fy")

    depth_f = _ensure_float(depth) * depth_scale
    if depth_f.ndim == 3:
        depth_f = depth_f[:, :, 0]

    h, w = depth_f.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float64),
                       np.arange(h, dtype=np.float64))

    mask = depth_f > 0
    z = depth_f[mask]
    x = (u[mask] - cx) * z / fx
    y = (v[mask] - cy) * z / fy

    xyz = np.column_stack([x, y, z])
    colors = None
    if color_image is not None:
        validate_image(color_image, "color_image")
        if color_image.ndim == 2:
            c = color_image[mask]
            colors = np.column_stack([c, c, c]).astype(np.uint8)
        else:
            colors = color_image[mask, :3].astype(np.uint8)

    logger.info("Generated point cloud with %d points from %dx%d depth map",
                len(xyz), w, h)
    return PointCloud(xyz=xyz, colors=colors)


# ====================================================================== #
#  Surface normal estimation                                              #
# ====================================================================== #


@log_operation(logger)
def estimate_normals(
    cloud: PointCloud,
    k: int = 20,
) -> PointCloud:
    """Estimate surface normals using PCA over *k* nearest neighbours.

    The resulting normals are oriented towards the camera (positive-z
    hemisphere).  Uses a brute-force kNN search suitable for clouds up
    to ~500k points; for larger clouds consider down-sampling first.

    Returns:
        A **new** :class:`PointCloud` with the ``normals`` field populated.
    """
    pts = cloud.xyz
    n = len(pts)
    if n < k:
        raise ValueError(f"Need at least k={k} points, got {n}")

    normals = np.empty_like(pts)

    # Use KD-tree for O(N log N) nearest-neighbour queries
    tree = cKDTree(pts)
    _, indices = tree.query(pts, k=k + 1)  # +1 because self is included
    knn_idx = indices[:, 1:]  # exclude self

    for i in range(n):
        neighbours = pts[knn_idx[i]]
        cov = np.cov(neighbours, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        normal = eigvecs[:, 0]  # smallest eigenvalue
        # Orient toward camera (positive z)
        if normal[2] < 0:
            normal = -normal
        normals[i] = normal

    return PointCloud(xyz=pts.copy(), colors=cloud.colors, normals=normals)


# ====================================================================== #
#  Plane fitting (RANSAC)                                                 #
# ====================================================================== #


@log_operation(logger)
def fit_plane_ransac(
    cloud: PointCloud,
    max_iterations: int = _DEFAULT_RANSAC_ITERATIONS,
    distance_threshold: float = _DEFAULT_RANSAC_THRESHOLD,
) -> PlaneModel:
    """Fit a plane to the point cloud using RANSAC.

    Parameters:
        cloud:              Input point cloud.
        max_iterations:     Number of RANSAC iterations.
        distance_threshold: Maximum distance for a point to be an inlier.

    Returns:
        A :class:`PlaneModel` with the best-fit plane and inlier mask.
    """
    pts = cloud.xyz
    n = len(pts)
    if n < 3:
        raise ValueError("Need at least 3 points to fit a plane")

    best_inlier_count = 0
    best_normal = np.array([0.0, 0.0, 1.0])
    best_d = 0.0
    best_mask = np.zeros(n, dtype=bool)

    rng = np.random.default_rng(42)
    for _ in range(max_iterations):
        idx = rng.choice(n, size=3, replace=False)
        p0, p1, p2 = pts[idx]
        normal = np.cross(p1 - p0, p2 - p0)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-12:
            continue
        normal /= norm_len
        d = -np.dot(normal, p0)

        distances = np.abs(pts @ normal + d)
        mask = distances < distance_threshold
        count = mask.sum()
        if count > best_inlier_count:
            best_inlier_count = count
            best_normal = normal
            best_d = d
            best_mask = mask

    # Refine with all inliers using least-squares
    inlier_pts = pts[best_mask]
    if len(inlier_pts) >= 3:
        centroid = inlier_pts.mean(axis=0)
        centered = inlier_pts - centroid
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        best_normal = vh[2]
        if best_normal[2] < 0:
            best_normal = -best_normal
        best_d = -np.dot(best_normal, centroid)
        distances = np.abs(pts @ best_normal + best_d)
        best_mask = distances < distance_threshold

    logger.info("RANSAC plane fit: %d / %d inliers (%.1f%%)",
                best_mask.sum(), n, 100.0 * best_mask.sum() / n)
    return PlaneModel(normal=best_normal, d=best_d, inliers=best_mask)


# ====================================================================== #
#  Height map extraction                                                  #
# ====================================================================== #


@log_operation(logger)
def extract_height_map(
    cloud: PointCloud,
    plane: PlaneModel,
) -> np.ndarray:
    """Compute signed height of each point above the fitted plane.

    Returns:
        (N,) float64 array of signed distances from the plane.
    """
    return cloud.xyz @ plane.normal + plane.d


# ====================================================================== #
#  Filtering                                                              #
# ====================================================================== #


@log_operation(logger)
def filter_statistical_outlier(
    cloud: PointCloud,
    k: int = _DEFAULT_STATISTICAL_K,
    std_ratio: float = _DEFAULT_STATISTICAL_STD,
) -> PointCloud:
    """Remove statistical outliers based on mean distance to *k* neighbours.

    Points whose mean kNN distance exceeds ``mean + std_ratio * std`` of
    all mean distances are removed.
    """
    pts = cloud.xyz
    n = len(pts)
    if n <= k:
        return cloud

    # Use KD-tree for O(N log N) nearest-neighbour queries
    tree = cKDTree(pts)
    distances, _ = tree.query(pts, k=k + 1)  # +1 because self is included
    mean_dists = distances[:, 1:].mean(axis=1)

    global_mean = mean_dists.mean()
    global_std = mean_dists.std()
    threshold = global_mean + std_ratio * global_std
    mask = mean_dists <= threshold

    logger.info("Statistical outlier filter: kept %d / %d points",
                mask.sum(), n)
    return _apply_mask(cloud, mask)


@log_operation(logger)
def filter_radius_outlier(
    cloud: PointCloud,
    radius: float = _DEFAULT_RADIUS,
    min_neighbours: int = _DEFAULT_MIN_NEIGHBOURS,
) -> PointCloud:
    """Remove points that have fewer than *min_neighbours* within *radius*."""
    pts = cloud.xyz
    n = len(pts)

    # Use KD-tree for O(N log N) radius queries
    tree = cKDTree(pts)
    neighbors = tree.query_ball_point(pts, r=radius)
    counts = np.array([len(nb) - 1 for nb in neighbors], dtype=np.int32)  # -1 for self

    mask = counts >= min_neighbours
    logger.info("Radius outlier filter (r=%.4f): kept %d / %d points",
                radius, mask.sum(), n)
    return _apply_mask(cloud, mask)


def _apply_mask(cloud: PointCloud, mask: np.ndarray) -> PointCloud:
    """Return a new cloud retaining only points where *mask* is True."""
    colors = cloud.colors[mask] if cloud.colors is not None else None
    normals = cloud.normals[mask] if cloud.normals is not None else None
    return PointCloud(xyz=cloud.xyz[mask], colors=colors, normals=normals)


# ====================================================================== #
#  Voxel grid down-sampling                                               #
# ====================================================================== #


@log_operation(logger)
def voxel_downsample(
    cloud: PointCloud,
    voxel_size: float = _DEFAULT_VOXEL_SIZE,
) -> PointCloud:
    """Down-sample the point cloud using a voxel grid.

    Each occupied voxel is represented by the centroid of its contained
    points.  Colours and normals are averaged accordingly.
    """
    validate_positive(voxel_size, "voxel_size")
    pts = cloud.xyz
    min_bound = pts.min(axis=0)
    keys = ((pts - min_bound) / voxel_size).astype(np.int64)

    # Unique voxel identification via Cantor-like hashing
    multiplier = np.array([1, keys[:, 0].max() + 1, 0], dtype=np.int64)
    multiplier[2] = multiplier[1] * (keys[:, 1].max() + 1)
    hashes = keys @ multiplier

    unique_hashes, inverse = np.unique(hashes, return_inverse=True)
    n_voxels = len(unique_hashes)

    new_xyz = np.zeros((n_voxels, 3), dtype=np.float64)
    np.add.at(new_xyz, inverse, pts)
    counts = np.bincount(inverse).astype(np.float64)[:, np.newaxis]
    new_xyz /= counts

    new_colors = None
    if cloud.colors is not None:
        new_colors = np.zeros((n_voxels, 3), dtype=np.float64)
        np.add.at(new_colors, inverse, cloud.colors.astype(np.float64))
        new_colors = (new_colors / counts).astype(np.uint8)

    new_normals = None
    if cloud.normals is not None:
        new_normals = np.zeros((n_voxels, 3), dtype=np.float64)
        np.add.at(new_normals, inverse, cloud.normals)
        norms = np.linalg.norm(new_normals, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        new_normals /= norms

    logger.info("Voxel downsample (size=%.4f): %d -> %d points",
                voxel_size, len(pts), n_voxels)
    return PointCloud(xyz=new_xyz, colors=new_colors, normals=new_normals)


# ====================================================================== #
#  3D distance computation                                                #
# ====================================================================== #


def point_to_point_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two 3D points."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b))


def cloud_to_cloud_distance(
    source: PointCloud,
    target: PointCloud,
) -> np.ndarray:
    """Compute per-point closest distance from *source* to *target*.

    Returns:
        (N,) float64 array of minimum distances for each source point.
    """
    src = source.xyz
    tgt = target.xyz
    n = len(src)
    min_dists = np.empty(n, dtype=np.float64)

    chunk = 2048
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        diff = tgt[np.newaxis, :, :] - src[start:end, np.newaxis, :]
        dists = np.sqrt(np.einsum("bnj,bnj->bn", diff, diff))
        min_dists[start:end] = dists.min(axis=1)

    return min_dists


# ====================================================================== #
#  Depth map visualisation                                                #
# ====================================================================== #


@log_operation(logger)
def visualise_depth(
    depth: np.ndarray,
    colormap: int = cv2.COLORMAP_JET,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> np.ndarray:
    """Convert a depth map to a colour-mapped BGR image for display.

    Parameters:
        depth:     (H, W) depth image.
        colormap:  OpenCV colormap constant (default ``COLORMAP_JET``).
        min_val:   Minimum depth for normalisation (auto if ``None``).
        max_val:   Maximum depth for normalisation (auto if ``None``).

    Returns:
        (H, W, 3) uint8 BGR colour image.
    """
    validate_image(depth, "depth")
    d = _ensure_float(depth)
    if d.ndim == 3:
        d = d[:, :, 0]

    valid = d > 0
    if min_val is None:
        min_val = float(d[valid].min()) if valid.any() else 0.0
    if max_val is None:
        max_val = float(d[valid].max()) if valid.any() else 1.0

    span = max_val - min_val
    if span < 1e-12:
        span = 1.0

    normalised = np.clip((d - min_val) / span, 0.0, 1.0)
    normalised_u8 = (normalised * 255).astype(np.uint8)
    # Set invalid pixels to zero
    normalised_u8[~valid] = 0

    coloured: np.ndarray = cv2.applyColorMap(normalised_u8, colormap)
    coloured[~valid] = 0
    return coloured


# ====================================================================== #
#  PLY I/O                                                                #
# ====================================================================== #


@log_operation(logger)
def save_ply(cloud: PointCloud, path: Union[str, Path]) -> None:
    """Save a point cloud to a binary little-endian PLY file."""
    path = Path(path)
    n = len(cloud)

    extra_lines: List[str] = []
    has_color = cloud.colors is not None
    has_normal = cloud.normals is not None
    if has_color:
        extra_lines += [
            "property uchar red\n",
            "property uchar green\n",
            "property uchar blue\n",
        ]
    if has_normal:
        extra_lines += [
            "property float nx\n",
            "property float ny\n",
            "property float nz\n",
        ]

    header = _PLY_HEADER_TEMPLATE.format(n=n, extra="".join(extra_lines))

    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        if not has_color and not has_normal:
            # Fast path: bulk write xyz only
            cloud.xyz.astype("<f4").tofile(fh)
        elif has_color and not has_normal:
            # Interleave xyz (3 floats) + rgb (3 bytes) per point
            xyz_f4 = cloud.xyz.astype("<f4")
            colors_rgb = cloud.colors[:, ::-1].astype(np.uint8)  # type: ignore[index]  # BGR->RGB
            for i in range(n):
                fh.write(xyz_f4[i].tobytes())
                fh.write(colors_rgb[i].tobytes())
        elif not has_color and has_normal:
            # Interleave xyz (3 floats) + normals (3 floats) per point
            xyz_f4 = cloud.xyz.astype("<f4")
            normals_f4 = cloud.normals.astype("<f4")  # type: ignore[union-attr]
            data = np.empty((n, 6), dtype="<f4")
            data[:, :3] = xyz_f4
            data[:, 3:] = normals_f4
            data.tofile(fh)
        else:
            # All fields: xyz + color + normal -- build structured array
            xyz_f4 = cloud.xyz.astype("<f4")
            colors_rgb = cloud.colors[:, ::-1].astype(np.uint8)  # type: ignore[index]  # BGR->RGB
            normals_f4 = cloud.normals.astype("<f4")  # type: ignore[union-attr]
            for i in range(n):
                fh.write(xyz_f4[i].tobytes())
                fh.write(colors_rgb[i].tobytes())
                fh.write(normals_f4[i].tobytes())

    logger.info("Saved %d points to %s", n, path)


@log_operation(logger)
def load_ply(path: Union[str, Path]) -> PointCloud:
    """Load a PLY file (binary little-endian or ASCII) into a PointCloud.

    Only ``x y z`` (and optionally ``red green blue``, ``nx ny nz``) properties
    are read.  Other properties are skipped.
    """
    path = Path(path)
    with open(path, "rb") as fh:
        # Parse header
        header_lines: List[str] = []
        while True:
            line = fh.readline().decode("ascii").strip()
            header_lines.append(line)
            if line == "end_header":
                break

        n_vertices = 0
        properties: List[Tuple[str, str]] = []
        for line in header_lines:
            if line.startswith("element vertex"):
                n_vertices = int(line.split()[-1])
            elif line.startswith("property"):
                parts = line.split()
                properties.append((parts[1], parts[2]))

        prop_names = [p[1] for p in properties]
        has_color = "red" in prop_names and "green" in prop_names
        has_normal = "nx" in prop_names

        # Build numpy structured dtype for bulk reading
        dtype_map = {"float": "<f4", "double": "<f8", "uchar": "u1",
                     "int": "<i4", "uint": "<u4", "short": "<i2", "ushort": "<u2"}
        dt_fields = [(name, dtype_map.get(ptype, "<f4")) for ptype, name in properties]
        record_dtype = np.dtype(dt_fields)

        raw = np.frombuffer(fh.read(n_vertices * record_dtype.itemsize), dtype=record_dtype)

    xyz = np.column_stack([raw["x"], raw["y"], raw["z"]]).astype(np.float64)
    colors = None
    if has_color:
        # PLY stores RGB; convert to BGR for PointCloud convention
        colors = np.column_stack([raw["blue"], raw["green"], raw["red"]]).astype(np.uint8)
    normals = None
    if has_normal:
        normals = np.column_stack([raw["nx"], raw["ny"], raw["nz"]]).astype(np.float64)

    logger.info("Loaded %d points from %s", n_vertices, path)
    return PointCloud(xyz=xyz, colors=colors, normals=normals)


# ====================================================================== #
#  Surface roughness & flatness                                           #
# ====================================================================== #


@log_operation(logger)
def measure_surface_roughness(
    cloud: PointCloud,
    plane: Optional[PlaneModel] = None,
    ransac_threshold: float = _DEFAULT_RANSAC_THRESHOLD,
) -> FlatnessResult:
    """Measure surface roughness relative to a reference plane.

    If *plane* is ``None`` a best-fit plane is computed automatically via
    RANSAC.

    Returns:
        A :class:`FlatnessResult` with RMS roughness, peak-to-valley, and
        per-point deviations.
    """
    if plane is None:
        plane = fit_plane_ransac(cloud, distance_threshold=ransac_threshold)

    deviations = cloud.xyz @ plane.normal + plane.d
    rms = float(np.sqrt(np.mean(deviations ** 2)))
    p2v = float(deviations.max() - deviations.min())
    mean_dev = float(np.mean(np.abs(deviations)))

    logger.info("Roughness: RMS=%.6f, PV=%.6f, mean=%.6f", rms, p2v, mean_dev)
    return FlatnessResult(
        rms_roughness=rms,
        peak_to_valley=p2v,
        mean_deviation=mean_dev,
        deviations=deviations,
    )


@log_operation(logger)
def measure_flatness(
    cloud: PointCloud,
    ransac_threshold: float = _DEFAULT_RANSAC_THRESHOLD,
) -> Dict[str, float]:
    """Convenience wrapper returning a flatness summary dictionary.

    Keys: ``rms``, ``peak_to_valley``, ``mean_deviation``, ``inlier_ratio``.
    """
    plane = fit_plane_ransac(cloud, distance_threshold=ransac_threshold)
    result = measure_surface_roughness(cloud, plane=plane)
    inlier_ratio = float(plane.inliers.sum()) / max(len(cloud), 1)
    return {
        "rms": result.rms_roughness,
        "peak_to_valley": result.peak_to_valley,
        "mean_deviation": result.mean_deviation,
        "inlier_ratio": inlier_ratio,
    }
