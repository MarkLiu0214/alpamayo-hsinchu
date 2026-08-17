#!/usr/bin/env python3
"""Extract the ROS1 bag once into an A-R1-friendly on-disk cache.

Outputs:
  <out>/cameras/<camera_key>/<timestamp_ns>.jpg
  <out>/lidar/<timestamp_ns>.npz
  <out>/poses.npz
  <out>/meta.json

Run inside a ROS Noetic environment, because this uses rosbag and sensor_msgs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import rosbag
import sensor_msgs.point_cloud2 as pc2

# Candidate semantic mapping for this particular bag. Validate visually before experiments.
CAMERA_TOPICS = {
    "cross_left": "/lucid_cameras_x01/gige_100_fl_hdr/compressed",
    "front_wide": "/lucid_cameras_x00/gige_100_f_hdr/compressed",
    "cross_right": "/lucid_cameras_x01/gige_100_fr_hdr/compressed",
    "front_tele": "/lucid_cameras_x00/gige_30_f_hdr/compressed",
}
POSE_TOPIC = "/local_pose"
LIDAR_TOPIC = "/ouster/top_122219002200"


def stamp_ns(msg, bag_time) -> int:
    """Prefer sensor acquisition time; fall back to bag-recording time."""
    header = getattr(msg, "header", None)
    if header is not None and header.stamp is not None and header.stamp.to_nsec() != 0:
        return int(header.stamp.to_nsec())
    return int(bag_time.to_nsec())


def save_compressed_image(msg, filename: Path) -> bool:
    encoded = np.frombuffer(msg.data, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        return False
    # BGR jpg is intentional: cv2.imread in the loader converts it back to RGB.
    return bool(cv2.imwrite(str(filename), image, [int(cv2.IMWRITE_JPEG_QUALITY), 95]))


def pointcloud2_to_xyz(msg) -> np.ndarray:
    # skip_nans removes invalid returns; return shape is guaranteed (N, 3).
    points = np.asarray(
        list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)),
        dtype=np.float32,
    )
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    return points.reshape(-1, 3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True, help="Path to a ROS1 .bag file")
    parser.add_argument("--out", required=True, help="Cache directory to create")
    parser.add_argument("--lidar-topic", default=LIDAR_TOPIC)
    parser.add_argument("--pose-topic", default=POSE_TOPIC)
    parser.add_argument("--max-lidar-points", type=int, default=120000,
                        help="Uniformly subsample larger clouds; 0 keeps all points")
    args = parser.parse_args()

    out = Path(args.out)
    camera_dirs = {key: out / "cameras" / key for key in CAMERA_TOPICS}
    for directory in [*camera_dirs.values(), out / "lidar"]:
        directory.mkdir(parents=True, exist_ok=True)

    reverse_camera_topics = {topic: key for key, topic in CAMERA_TOPICS.items()}
    topics = [*CAMERA_TOPICS.values(), args.pose_topic, args.lidar_topic]
    pose_t, pose_xyz, pose_quat_xyzw = [], [], []
    image_count = {key: 0 for key in CAMERA_TOPICS}
    lidar_count = 0

    with rosbag.Bag(args.bag, "r") as bag:
        bag_start_ns = int(bag.get_start_time() * 1e9)
        bag_end_ns = int(bag.get_end_time() * 1e9)
        for topic, msg, bag_time in bag.read_messages(topics=topics):
            t_ns = stamp_ns(msg, bag_time)
            if topic in reverse_camera_topics:
                key = reverse_camera_topics[topic]
                filename = camera_dirs[key] / f"{t_ns}.jpg"
                if save_compressed_image(msg, filename):
                    image_count[key] += 1

            elif topic == args.pose_topic:
                p = msg.pose.position
                q = msg.pose.orientation
                pose_t.append(t_ns)
                pose_xyz.append((p.x, p.y, p.z))
                pose_quat_xyzw.append((q.x, q.y, q.z, q.w))

            elif topic == args.lidar_topic:
                xyz = pointcloud2_to_xyz(msg)
                if args.max_lidar_points > 0 and len(xyz) > args.max_lidar_points:
                    indices = np.linspace(0, len(xyz) - 1, args.max_lidar_points).astype(np.int64)
                    xyz = xyz[indices]
                np.savez_compressed(out / "lidar" / f"{t_ns}.npz", xyz=xyz)
                lidar_count += 1

    if not pose_t:
        raise RuntimeError(f"No pose received from {args.pose_topic}")
    order = np.argsort(np.asarray(pose_t, dtype=np.int64))
    np.savez_compressed(
        out / "poses.npz",
        t_ns=np.asarray(pose_t, dtype=np.int64)[order],
        xyz=np.asarray(pose_xyz, dtype=np.float64)[order],
        quat_xyzw=np.asarray(pose_quat_xyzw, dtype=np.float64)[order],
    )
    meta = {
        "bag": str(Path(args.bag).resolve()),
        "bag_start_ns": bag_start_ns,
        "bag_end_ns": bag_end_ns,
        "camera_topics": CAMERA_TOPICS,
        "pose_topic": args.pose_topic,
        "lidar_topic": args.lidar_topic,
        "image_count": image_count,
        "pose_count": len(pose_t),
        "lidar_count": lidar_count,
        "NOTE": "LiDAR xyz are still in the PointCloud2 frame. Fill T_EGO_FROM_LIDAR in load_rosbag_with_lidar.py from /tf_static before overlaying A-R1 trajectories.",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
