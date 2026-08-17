#!/usr/bin/env python3
"""Load a pre-extracted ROS1 bag cache in the dictionary format used by Alpamayo-R1."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp

# A-R1 camera token ordering used by the Physical AI AV loader.
CAMERA_KEYS = ("cross_left", "front_wide", "cross_right", "front_tele")
CAMERA_INDICES = torch.tensor([0, 1, 2, 6], dtype=torch.int64)

# Replace this 4x4 matrix after inspecting /tf_static.
# It maps a point represented in the selected LiDAR frame to the vehicle ego/base_link frame.
# Identity is valid ONLY when the PointCloud2 frame_id already is the ego/base_link frame.
T_EGO_FROM_LIDAR = np.eye(4, dtype=np.float64)

T_EGO_FROM_LIDAR[:3, :3] = Rotation.from_quat(
    [-0.0, 0.0051, 0.0041, 1.0]
).as_matrix()


def _nearest_index(times_ns: np.ndarray, requested_ns: int) -> int:
    i = int(np.searchsorted(times_ns, requested_ns))
    candidates = [max(i - 1, 0), min(i, len(times_ns) - 1)]
    return min(candidates, key=lambda j: abs(int(times_ns[j]) - requested_ns))


def _read_times(directory: Path, suffix: str) -> np.ndarray:
    files = sorted(directory.glob(f"*{suffix}"), key=lambda p: int(p.stem))
    if not files:
        raise FileNotFoundError(f"No {suffix} files in {directory}")
    return np.asarray([int(p.stem) for p in files], dtype=np.int64)


class RosbagAR1Cache:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        poses = np.load(self.root / "poses.npz")
        self.pose_t = poses["t_ns"].astype(np.int64)
        self.pose_xyz = poses["xyz"].astype(np.float64)
        self.pose_rot = Rotation.from_quat(poses["quat_xyzw"].astype(np.float64))
        self.camera_t = {
            key: _read_times(self.root / "cameras" / key, ".jpg") for key in CAMERA_KEYS
        }
        self.lidar_t = _read_times(self.root / "lidar", ".npz")

    def valid_t0_range_ns(self, num_history_steps=16, time_step_s=0.1, num_frames=4):
        earliest_offset_ns = max((num_history_steps - 1) * time_step_s, (num_frames - 1) * time_step_s) * 1e9
        return int(self.pose_t[0] + earliest_offset_ns), int(self.pose_t[-1])

    def pose_at(self, timestamps_ns: np.ndarray) -> tuple[np.ndarray, Rotation]:
        timestamps_ns = np.asarray(timestamps_ns, dtype=np.int64)
        if timestamps_ns.min() < self.pose_t[0] or timestamps_ns.max() > self.pose_t[-1]:
            raise ValueError("Requested history extends outside /local_pose time range")
        xyz = np.column_stack([
            np.interp(timestamps_ns, self.pose_t, self.pose_xyz[:, axis]) for axis in range(3)
        ])
        # Slerp needs seconds or any monotonically increasing numeric scale.
        rotations = Slerp(self.pose_t.astype(np.float64), self.pose_rot)(timestamps_ns.astype(np.float64))
        return xyz, rotations

    @lru_cache(maxsize=256)
    def image_at(self, key: str, requested_ns: int) -> tuple[np.ndarray, int]:
        ts = self.camera_t[key]
        actual_ns = int(ts[_nearest_index(ts, requested_ns)])
        image_bgr = cv2.imread(str(self.root / "cameras" / key / f"{actual_ns}.jpg"), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Cannot decode {key} image at {actual_ns}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return image_rgb, actual_ns

    @lru_cache(maxsize=64)
    def lidar_at(self, requested_ns: int) -> tuple[np.ndarray, int]:
        actual_ns = int(self.lidar_t[_nearest_index(self.lidar_t, requested_ns)])
        xyz_lidar = np.load(self.root / "lidar" / f"{actual_ns}.npz")["xyz"].astype(np.float64)
        hom = np.column_stack([xyz_lidar, np.ones(len(xyz_lidar))])
        xyz_ego = (T_EGO_FROM_LIDAR @ hom.T).T[:, :3]
        return xyz_ego.astype(np.float32), actual_ns


def load_rosbag_with_lidar(
    cache: RosbagAR1Cache,
    t0_ns: int,
    num_history_steps: int = 16,
    time_step: float = 0.1,
    num_frames: int = 4,
) -> dict[str, Any]:
    """Return exactly the fields consumed by your current inference code.

    t0 is the newest image / pose instant. It uses 16 ego poses from t0-1.5s to t0
    and four images from t0-0.3s to t0. The model does not consume future GT.
    """
    dt_ns = int(round(time_step * 1e9))
    history_t = t0_ns + np.arange(-(num_history_steps - 1), 1, dtype=np.int64) * dt_ns
    history_xyz_world, history_rot_world = cache.pose_at(history_t)
    t0_xyz = history_xyz_world[-1]
    t0_rot = history_rot_world[-1]
    t0_rot_inv = t0_rot.inv()

    history_xyz_ego = t0_rot_inv.apply(history_xyz_world - t0_xyz)
    history_rot_ego = (t0_rot_inv * history_rot_world).as_matrix()

    image_t = t0_ns + np.arange(-(num_frames - 1), 1, dtype=np.int64) * dt_ns
    frames_by_camera, timestamps_by_camera = [], []
    for key in CAMERA_KEYS:
        frames, actual_times = [], []
        for t in image_t:
            frame, actual_t = cache.image_at(key, int(t))
            frames.append(frame)
            actual_times.append(actual_t)
        arr = np.stack(frames, axis=0)  # T,H,W,3
        frames_by_camera.append(torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous())
        timestamps_by_camera.append(torch.tensor(actual_times, dtype=torch.int64))

    image_frames = torch.stack(frames_by_camera, dim=0)  # 4,T,3,H,W
    absolute_timestamps = torch.stack(timestamps_by_camera, dim=0)
    relative_timestamps = (absolute_timestamps - absolute_timestamps.min()).float() * 1e-9
    point_cloud, lidar_t_ns = cache.lidar_at(int(t0_ns))

    return {
        "image_frames": image_frames,
        "camera_indices": CAMERA_INDICES.clone(),
        "ego_history_xyz": torch.from_numpy(history_xyz_ego).float()[None, None],
        "ego_history_rot": torch.from_numpy(history_rot_ego).float()[None, None],
        "relative_timestamps": relative_timestamps,
        "absolute_timestamps": absolute_timestamps,
        "point_cloud": point_cloud,
        "t0_ns": int(t0_ns),
        "lidar_t_ns": int(lidar_t_ns),
    }
