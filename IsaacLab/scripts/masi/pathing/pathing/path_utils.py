"""
Path utilities for planning and execution.

Contains:
- Path interpolation and analysis functions
- Quaternion and rotation utilities
- Obstacle checking
- Grid configuration
- Workspace bounds calculation

All algorithms use these shared utilities for consistency.
"""

from typing import Any, Tuple, List, Dict, Optional
from dataclasses import dataclass
import numpy as np


def interpolate_path(path: np.ndarray, interpolation_factor: int) -> np.ndarray:
    """Linearly densify a waypoint path by an integer interpolation factor.

    Keeps endpoints fixed, removes consecutive duplicate points, and returns
    a float32 NumPy array for downstream numeric use.
    """
    if path is None:
        return path
    pts = np.asarray(path, dtype=np.float32)
    n = len(pts)
    if n < 2:
        return pts

    dedup = [pts[0]]
    for i in range(1, n):
        if not np.allclose(pts[i], dedup[-1]):
            dedup.append(pts[i])
    pts = np.asarray(dedup, dtype=np.float32)
    if len(pts) < 2:
        return pts

    seg = pts[1:] - pts[:-1]
    seg_len = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len, dtype=np.float32)])
    total_len = cum[-1]
    if total_len == 0.0:
        return pts[:1]

    target_samples = int((len(pts) - 1) * (interpolation_factor + 1) + 1)
    target_samples = max(2, target_samples)

    s_samples = np.linspace(0.0, float(total_len), target_samples, dtype=np.float32)

    def find_seg(s):
        i = np.searchsorted(cum, s, side="right") - 1
        return min(max(i, 0), len(seg_len) - 1)

    out = []
    for s in s_samples:
        i = find_seg(s)
        s0, s1 = float(cum[i]), float(cum[i + 1])
        t = 0.0 if s1 == s0 else (float(s) - s0) / (s1 - s0)
        P = pts[i] + t * (pts[i + 1] - pts[i])
        out.append(P.astype(np.float32))

    out[0] = pts[0]
    out[-1] = pts[-1]

    return np.asarray(out, dtype=np.float32)


def calculate_path_length(path: np.ndarray) -> float:
    """
    Calculate total length of path in meters.
    
    Args:
        path: Path waypoints as np.ndarray of shape [N, 3]
    Returns:
        Total path length in meters
    """
    if len(path) < 2:
        return 0.0

    total_length = 0.0
    for i in range(len(path) - 1):
        segment_length = np.linalg.norm(path[i + 1] - path[i])
        total_length += segment_length

    return total_length


def interpolate_along_path(path: np.ndarray, progress: float) -> np.ndarray:
    """
    Get position along path based on progress (0.0 to 1.0). Mainly for IRL debugging

    Args:
        path: Path waypoints as np.ndarray of shape [N, 3]
        progress: Progress along path from 0.0 (start) to 1.0 (end)
    Returns:
        Interpolated position as np.ndarray of shape [3]
    """
    if progress <= 0.0:
        return path[0]
    if progress >= 1.0:
        return path[-1]

    # Calculate cumulative distances
    distances = [0.0]
    for i in range(len(path) - 1):
        segment_length = np.linalg.norm(path[i + 1] - path[i])
        distances.append(distances[-1] + segment_length)

    total_length = distances[-1]
    if total_length == 0:
        return path[0]

    target_distance = progress * total_length

    # Find which segment we're in
    for i in range(len(distances) - 1):
        if target_distance <= distances[i + 1]:
            # Interpolate within this segment
            segment_progress = (target_distance - distances[i]) / (distances[i + 1] - distances[i])
            return path[i] + segment_progress * (path[i + 1] - path[i])

    return path[-1]


def precompute_cumulative_distances(path: np.ndarray) -> Tuple[np.ndarray, float]:
    """Precompute cumulative arc-length distances for a polyline path.

    Returns (cum_dist, total_length) where cum_dist[i] is the distance from
    the start to waypoint i.
    """
    pts = np.asarray(path, dtype=np.float32)
    if len(pts) == 0:
        return np.asarray([0.0], dtype=np.float32), 0.0
    if len(pts) == 1:
        return np.asarray([0.0], dtype=np.float32), 0.0

    seg = pts[1:] - pts[:-1]
    seg_len = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len, dtype=np.float32)])
    total_len = float(cum[-1])
    return cum.astype(np.float32), total_len


def interpolate_at_s(path: np.ndarray, cum_dist: np.ndarray, s: float) -> np.ndarray:
    """Interpolate a point along the path at absolute arc length s (meters)."""
    pts = np.asarray(path, dtype=np.float32)
    if len(pts) == 0:
        return np.zeros(3, dtype=np.float32)
    if len(pts) == 1:
        return pts[0]

    total = float(cum_dist[-1])
    if s <= 0.0:
        return pts[0]
    if s >= total:
        return pts[-1]

    i = int(np.searchsorted(cum_dist, s, side="right") - 1)
    i = min(max(i, 0), len(pts) - 2)
    s0 = float(cum_dist[i])
    s1 = float(cum_dist[i + 1])
    t = 0.0 if s1 == s0 else (float(s) - s0) / (s1 - s0)
    return (pts[i] + t * (pts[i + 1] - pts[i])).astype(np.float32)


# ============================================================================
# Path normalization (constant-speed resampling and waypoint reduction)
# ============================================================================

def resample_constant_speed(path: np.ndarray, speed_mps: float, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """Resample a polyline path to constant-speed samples.

    Returns (positions, times), where:
      - positions: np.ndarray [K, 3] float32 sampled at equal arc-length steps s_step = speed_mps * dt
      - times: np.ndarray [K] float32, starting at 0.0, matching positions
    """
    pts = np.asarray(path, dtype=np.float32)
    if len(pts) < 2 or speed_mps <= 0.0 or dt <= 0.0:
        return pts, np.asarray([0.0], dtype=np.float32)

    cum_dist, total_len = precompute_cumulative_distances(pts)
    s_step = float(speed_mps * dt)
    if s_step <= 0.0 or total_len <= 0.0:
        return pts, np.asarray([0.0], dtype=np.float32)

    n_steps = int(np.ceil(total_len / s_step)) + 1
    s_vals = np.linspace(0.0, float(total_len), n_steps, dtype=np.float32)
    positions = np.stack([interpolate_at_s(pts, cum_dist, float(s)) for s in s_vals], axis=0)
    times = s_vals / float(speed_mps)
    return positions.astype(np.float32), times.astype(np.float32)


def resample_constant_circum_speed(
    path: np.ndarray,
    *,
    speed_mps: float,
    dt: float,
    center_xz: Tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Resample a path to approximate constant circumferential speed about the base.

    Tailored for the excavator-style setup where the robot slews around a
    vertical axis through the origin and motion is primarily in the X-Z plane.
    It treats each waypoint as (x, y, z) in base/world frame and computes polar
    coordinates in the X-Z plane:

    - radius r = sqrt(x^2 + z^2)
    - angle  theta = atan2(z, x)

    The path is then reparameterized so that the cumulative circumferential
    arc length r * |Δtheta| grows approximately linearly along the trajectory.
    Geometric resampling is done before the standardizer; actual execution
    timing is still governed by ``standardize_path`` (constant linear speed).

    Args:
        path: Input waypoints, shape [N, 3] (x, y, z) in base/world frame.
        speed_mps: Target circumferential speed (m/s) used to choose the
            approximate sampling density. Typically the same as the
            standardizer's ``speed_mps``.
        dt: Approximate sample period (s). Used together with ``speed_mps``
            to decide how finely to resample. The final uniform timing still
            comes from the standardizer.
        center_xz: (x, z) coordinates of the rotation axis. For the current
            demos this is always (0, 0).

    Returns:
        Resampled path of shape [M, 3]. If the path has fewer than 2 points or
        the total circumferential length is negligible, the input path is
        returned unchanged.
    """
    pts = np.asarray(path, dtype=np.float32)
    if pts.shape[0] < 2:
        return pts

    if speed_mps <= 0.0 or dt <= 0.0:
        # Degenerate config: nothing sensible to do, just return original.
        return pts

    cx, cz = center_xz

    # Compute radius and angle in X-Z plane for each point.
    x = pts[:, 0] - cx
    z = pts[:, 2] - cz
    r = np.sqrt(x * x + z * z).astype(np.float32)

    # Avoid division by zero for points exactly on the axis — they contribute
    # no circumferential motion anyway, so treat them as having a tiny radius.
    r_safe = np.where(r < 1e-6, 1e-6, r)

    theta = np.arctan2(z, x).astype(np.float32)
    theta_unwrapped = np.unwrap(theta.astype(np.float64)).astype(np.float32)

    # Segment-wise circumferential arc length: r_avg * |Δtheta|
    dtheta = np.diff(theta_unwrapped)
    r_mid = 0.5 * (r_safe[:-1] + r_safe[1:])
    arc_seg = np.abs(r_mid * dtheta).astype(np.float32)

    total_arc = float(np.sum(arc_seg))
    if total_arc < 1e-6:
        # Path has effectively no circumferential extent.
        return pts

    # Choose number of samples so that average arc between samples is close
    # to speed_mps * dt. The standardizer will later enforce exact timing.
    target_arc = speed_mps * dt
    if target_arc <= 0.0:
        return pts

    num_segments = max(int(np.ceil(total_arc / target_arc)), 1)
    num_samples = num_segments + 1

    arc_cum = np.concatenate([[0.0], np.cumsum(arc_seg, dtype=np.float32)])
    s_targets = np.linspace(0.0, total_arc, num_samples, dtype=np.float32)

    resampled: List[np.ndarray] = []
    seg_idx = 0
    for s in s_targets:
        # Find segment containing this arc position.
        while seg_idx + 1 < len(arc_cum) and s > arc_cum[seg_idx + 1]:
            seg_idx += 1

        if seg_idx >= len(arc_seg):
            # Numerical edge-case: clamp to last point.
            resampled.append(pts[-1])
            continue

        s0 = arc_cum[seg_idx]
        s1 = arc_cum[seg_idx + 1]
        if s1 <= s0:
            alpha = 0.0
        else:
            alpha = float((s - s0) / (s1 - s0))

        p = (1.0 - alpha) * pts[seg_idx] + alpha * pts[seg_idx + 1]
        resampled.append(p.astype(np.float32))

    return np.stack(resampled, axis=0)


def downsample_by_points(path: np.ndarray, max_points: int) -> np.ndarray:
    """Reduce a path to at most `max_points` waypoints, spaced evenly by arc length.

    Keeps endpoints fixed. Returns float32 array [M, 3] where M <= max_points.
    """
    pts = np.asarray(path, dtype=np.float32)
    if max_points is None or max_points <= 0 or len(pts) <= 2 or max_points >= len(pts):
        return pts

    max_points = max(2, int(max_points))
    cum_dist, total_len = precompute_cumulative_distances(pts)
    if total_len <= 0.0:
        return pts[:1]

    s_vals = np.linspace(0.0, float(total_len), max_points, dtype=np.float32)
    out = np.stack([interpolate_at_s(pts, cum_dist, float(s)) for s in s_vals], axis=0)
    out[0] = pts[0]
    out[-1] = pts[-1]
    return out.astype(np.float32)


def _identity_quaternions(n: int) -> np.ndarray:
    """Return an array of n identity quaternions [w,x,y,z] as float32."""
    if n <= 0:
        return np.zeros((0, 4), dtype=np.float32)
    q = np.zeros((n, 4), dtype=np.float32)
    q[:, 0] = 1.0  # w = 1
    return q


def build_poses_xyz_quat(positions: np.ndarray) -> np.ndarray:
    """Build Nx7 pose array [x,y,z,qw,qx,qy,qz] from Nx3 positions using identity quaternions.

    IK-friendly, lightweight, and robust against singularities.
    """
    pos = np.asarray(positions, dtype=np.float32)
    n = len(pos)
    quats = _identity_quaternions(n)
    if n == 0:
        return np.zeros((0, 7), dtype=np.float32)
    poses = np.concatenate([pos, quats], axis=1)
    return poses.astype(np.float32)


def standardize_path(path: np.ndarray,
                     speed_mps: float,
                     dt: float,
                     return_poses: bool = True) -> Dict[str, np.ndarray]:
    """Create a normalized, constant-speed representation of a path.

    Args:
        path: Raw path [N,3]
        speed_mps: Target constant speed (m/s)
        dt: Sample period for execution (s)
        return_poses: If True, include Nx7 pose arrays with identity quaternions

    Returns dict with:
        - exec_positions: [K,3] constant-speed samples
        - times: [K] timestamps (s)
        - total_length_m: [1] total path length (m)
        - duration_s: [1] total duration (s)
        - exec_poses: [K,7] (optional)
    """
    pts = np.asarray(path, dtype=np.float32)
    exec_positions, times = resample_constant_speed(pts, speed_mps, dt)

    result: Dict[str, np.ndarray] = {
        "exec_positions": exec_positions,
        "times": times,
        "total_length_m": np.asarray([calculate_path_length(exec_positions)], dtype=np.float32),
        "duration_s": np.asarray([times[-1] if len(times) else 0.0], dtype=np.float32),
    }

    if return_poses:
        result["exec_poses"] = build_poses_xyz_quat(exec_positions)

    return result


def project_point_onto_path(point: np.ndarray, path: np.ndarray, cum_dist: np.ndarray) -> Tuple[float, np.ndarray, int, float]:
    """Project a 3D point onto a polyline path.

    Returns:
        s_proj: Arc length (meters) to the projection point from start
        p_proj: 3D coordinates of the projection
        seg_idx: Index of segment used for projection (start waypoint index)
        t: Segment-local interpolation factor in [0, 1]
    """
    p = np.asarray(point, dtype=np.float32)
    pts = np.asarray(path, dtype=np.float32)
    if len(pts) == 0:
        return 0.0, np.zeros(3, dtype=np.float32), 0, 0.0
    if len(pts) == 1:
        return 0.0, pts[0], 0, 0.0

    best_dist2 = float("inf")
    best_s = 0.0
    best_proj = pts[0]
    best_i = 0
    best_t = 0.0

    for i in range(len(pts) - 1):
        a = pts[i]
        b = pts[i + 1]
        ab = b - a
        ab_len2 = float(np.dot(ab, ab))
        if ab_len2 <= 1e-12:
            # Degenerate segment; treat as point
            t = 0.0
            proj = a
            s_here = float(cum_dist[i])
        else:
            t = float(np.dot(p - a, ab) / ab_len2)
            t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
            proj = a + t * ab
            seg_len = float(np.sqrt(ab_len2))
            s_here = float(cum_dist[i]) + t * seg_len

        d2 = float(np.dot(p - proj, p - proj))
        if d2 < best_dist2:
            best_dist2 = d2
            best_s = s_here
            best_proj = proj
            best_i = i
            best_t = t

    return best_s, best_proj.astype(np.float32), best_i, float(best_t)


def calculate_execution_time(path: np.ndarray, speed_mps: float) -> float:
    """
    Calculate estimated execution time for a path.

    Args:
        path: Path waypoints as np.ndarray of shape [N, 3]
        speed_mps: Speed in meters per second
    Returns:
        Estimated execution time in seconds
    """
    total_distance = calculate_path_length(path)
    return total_distance / speed_mps if speed_mps > 0 else 0.0


def is_target_reached(current_pos: np.ndarray, target_pos: np.ndarray, config: Any) -> bool:
    """
    Check if target position is reached within tolerance.

    Args:
        current_pos: Current position [x, y, z]
        target_pos: Target position [x, y, z]
        config: Object with attribute `final_target_tolerance`
    Returns:
        True if target is reached within tolerance
    """
    distance = np.linalg.norm(current_pos - target_pos)
    return distance < getattr(config, "final_target_tolerance", 0.0)


def print_path_info(path: np.ndarray, config: Any, label: str = "Path") -> None:
    """
    Print detailed path information for debugging.

    Args:
        path: Path waypoints
        config: Object with attribute `speed_mps`
        label: Label for the path (e.g., "A* Path", "Smooth Path")
    """
    length = calculate_path_length(path)
    estimated_time = calculate_execution_time(path, getattr(config, "speed_mps", 0.0))
    avg_spacing = length / (len(path) - 1) if len(path) > 1 else 0

    print(f"[{label}] {len(path)} waypoints, {length:.3f}m total")
    print(f"[{label}] Speed: {getattr(config, 'speed_mps', 0.0):.3f}m/s, Est. time: {estimated_time:.1f}s")
    print(f"[{label}] Avg spacing: {avg_spacing:.4f}m")


# ============================================================================
# Quaternion and Rotation Utilities
# ============================================================================

def normalize_vector(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Normalize a vector to unit length."""
    n = float(np.linalg.norm(v))
    return v / max(n, eps)


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = q

    # Normalize quaternion
    norm = np.sqrt(w*w + x*x + y*y + z*z)
    if norm == 0:
        return np.eye(3)
    w, x, y, z = w/norm, x/norm, y/norm, z/norm

    # Rotation matrix from quaternion
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])

    return R


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion [w,x,y,z]."""
    m00, m01, m02 = R[0,0], R[0,1], R[0,2]
    m10, m11, m12 = R[1,0], R[1,1], R[1,2]
    m20, m21, m22 = R[2,0], R[2,1], R[2,2]
    t = m00 + m11 + m22
    if t > 0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif (m00 > m11) and (m00 > m22):
        s = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    return np.array([w, x, y, z], dtype=np.float32)


def quaternion_conjugate(q: np.ndarray) -> np.ndarray:
    """Return the conjugate of a quaternion [w,x,y,z]."""
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=np.float32)


def basis_start_goal_plane(start_w: np.ndarray, goal_w: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build an orthonormal basis for the vertical plane through start→goal:
    Xp = along start→goal, Zp = in plane (mostly 'up'), Yp = plane normal.
    """
    v = goal_w - start_w
    Xp = normalize_vector(v)
    z_world = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    # Plane normal Yp = Xp × z
    Yp = np.cross(Xp, z_world)
    if np.linalg.norm(Yp) < 1e-6:
        # Degenerate (start→goal nearly vertical). Use X axis instead of z to define a stable plane.
        x_world = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        Yp = np.cross(Xp, x_world)
        if np.linalg.norm(Yp) < 1e-6:
            # As a last resort, fall back to Y axis
            y_world = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            Yp = np.cross(Xp, y_world)
    Yp = normalize_vector(Yp)

    # In-plane vertical-like axis Zp = Yp × Xp
    Zp = normalize_vector(np.cross(Yp, Xp))
    return Xp, Yp, Zp  # columns for rotation matrix R = [Xp Yp Zp]


# ============================================================================
# Workspace Bounds Calculation
# ============================================================================

def calculate_workspace_bounds(obstacle_data: List[Dict[str, Any]],
                               start_pos: Tuple[float, float, float],
                               goal_pos: Tuple[float, float, float],
                               padding: float = 0.1,
                               min_bounds: Tuple[float, float, float] = (0.34, -0.36, 0.0),
                               max_bounds: Tuple[float, float, float] = (0.85, 0.36, 0.78)) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Calculate workspace bounds based on obstacles, start, and goal positions.

    Used by all path planning algorithms to ensure consistent workspace sizing.

    Args:
        obstacle_data: List of obstacle dictionaries with "pos" keys
        start_pos: Start position (x, y, z)
        goal_pos: Goal position (x, y, z)
        padding: Extra space around obstacles/points
        min_bounds: Minimum workspace bounds (enforced)
        max_bounds: Maximum workspace bounds (enforced)

    Returns:
        (bounds_min, bounds_max) as tuples
    """
    if len(obstacle_data) > 0:
        all_positions = [obs["pos"] for obs in obstacle_data] + [list(start_pos), list(goal_pos)]
        all_positions = np.array(all_positions)

        bounds_min_calc = np.min(all_positions, axis=0) - padding
        bounds_max_calc = np.max(all_positions, axis=0) + padding

        # Ensure minimum workspace size
        bounds_min_calc = np.minimum(bounds_min_calc, min_bounds)
        bounds_max_calc = np.maximum(bounds_max_calc, max_bounds)

        return tuple(bounds_min_calc), tuple(bounds_max_calc)
    else:
        # Default bounds
        return min_bounds, max_bounds


# ============================================================================
# Grid Configuration
# ============================================================================

@dataclass
class GridConfig:
    """Configuration for the path planning grid/workspace."""
    resolution: float = 0.01  # Grid cell size in meters (for A*)
    bounds_min: Tuple[float, float, float] = (0.0, -1.0, 0.1)  # (x_min, y_min, z_min)
    bounds_max: Tuple[float, float, float] = (1.0, 1.0, 2.0)   # (x_max, y_max, z_max)
    safety_margin: float = 0.02  # Additional clearance around obstacles


# ============================================================================
# Obstacle Checking
# ============================================================================

class ObstacleChecker:
    """Handles obstacle collision detection for path planning."""

    def __init__(self, obstacle_data: List[Dict[str, Any]], safety_margin: float = 0.02):
        """
        Initialize obstacle checker.

        Args:
            obstacle_data: List of obstacle dictionaries with keys:
                - "size": np.array([x, y, z]) - obstacle dimensions
                - "pos" or "position": np.array([x, y, z]) - obstacle center position
                - "rot" or "rotation": np.array([w, x, y, z]) - quaternion rotation
            safety_margin: Additional clearance around obstacles
        """
        self.obstacles = []
        self.safety_margin = safety_margin

        # Normalize obstacle data format
        for obs in obstacle_data:
            size = np.asarray(obs.get("size", [0.1, 0.1, 0.1]), dtype=np.float32)
            # Apply symmetric padding once up front (allow slight negative for debugging but clamp)
            size = size + self.safety_margin * 2
            size = np.maximum(size, 1e-4)

            normalized = {
                "size": size,
                "pos": np.asarray(obs.get("pos", obs.get("position", [0, 0, 0])), dtype=np.float32),
                "rot": np.asarray(obs.get("rot", obs.get("rotation", [1, 0, 0, 0])), dtype=np.float32),
            }
            self.obstacles.append(normalized)

        # Pre-compute axis-aligned bounding boxes for efficiency
        self._compute_aabbs()

    def _compute_aabbs(self):
        """Pre-compute axis-aligned bounding boxes for all obstacles."""
        self.aabbs = []

        for obs in self.obstacles:
            size = obs["size"]
            pos = obs["pos"]

            # Compute oriented bounding box corners (handles rotated obstacles)
            corners = self._get_box_corners(size, pos, obs["rot"])
            min_bounds = np.min(corners, axis=0)
            max_bounds = np.max(corners, axis=0)

            self.aabbs.append({
                "min": min_bounds,
                "max": max_bounds,
                "detailed": obs  # Keep reference for detailed collision checking
            })

    def _get_box_corners(self, size: np.ndarray, position: np.ndarray,
                        quaternion: np.ndarray) -> np.ndarray:
        """Get the 8 corners of a rotated box."""
        # Create local corners (before rotation)
        half_size = size / 2
        local_corners = np.array([
            [-half_size[0], -half_size[1], -half_size[2]],
            [+half_size[0], -half_size[1], -half_size[2]],
            [-half_size[0], +half_size[1], -half_size[2]],
            [+half_size[0], +half_size[1], -half_size[2]],
            [-half_size[0], -half_size[1], +half_size[2]],
            [+half_size[0], -half_size[1], +half_size[2]],
            [-half_size[0], +half_size[1], +half_size[2]],
            [+half_size[0], +half_size[1], +half_size[2]],
        ])

        # Convert quaternion to rotation matrix
        rot_matrix = quaternion_to_rotation_matrix(quaternion)

        # Apply rotation and translation
        world_corners = np.array([rot_matrix @ corner + position for corner in local_corners])

        return world_corners

    def is_point_collision_free(self, point: Tuple[float, float, float]) -> bool:
        """Check if a point is collision-free."""
        point_array = np.array(point)

        for aabb in self.aabbs:
            # Quick AABB check first
            if np.all(point_array >= aabb["min"]) and np.all(point_array <= aabb["max"]):
                # Detailed check for rotated obstacles
                if not self._point_in_oriented_box(point_array, aabb["detailed"]):
                    continue
                return False

        return True

    def _point_in_oriented_box(self, point: np.ndarray, obstacle: Dict[str, Any]) -> bool:
        """Check if point is inside an oriented bounding box."""
        # For axis-aligned boxes (no rotation), use simple bounds check
        if np.allclose(obstacle["rot"], [1, 0, 0, 0], atol=1e-3):
            half_size = obstacle["size"] / 2
            min_bounds = obstacle["pos"] - half_size
            max_bounds = obstacle["pos"] + half_size
            return np.all(point >= min_bounds) and np.all(point <= max_bounds)

        # For rotated boxes, transform point to local coordinates
        rot_matrix = quaternion_to_rotation_matrix(obstacle["rot"])
        local_point = rot_matrix.T @ (point - obstacle["pos"])
        half_size = obstacle["size"] / 2

        return np.all(np.abs(local_point) <= half_size)

    def is_line_collision_free(self, start: Tuple[float, float, float],
                              end: Tuple[float, float, float],
                              num_samples: int = 10) -> bool:
        """Check if a line segment is collision-free by sampling points."""
        start_array = np.array(start)
        end_array = np.array(end)

        for i in range(num_samples + 1):
            t = i / num_samples
            sample_point = start_array + t * (end_array - start_array)
            if not self.is_point_collision_free(tuple(sample_point)):
                return False

        return True
