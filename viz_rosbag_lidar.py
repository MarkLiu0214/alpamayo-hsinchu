#!/usr/bin/env python3
"""Render A-R1 predictions over cached ROS LiDAR points."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


def make_trajectory(xyz: np.ndarray, radius: float = 0.20):
    """Create red spheres along the predicted trajectory."""
    mesh = o3d.geometry.TriangleMesh()

    for point in xyz:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.translate(point)
        mesh += sphere

    mesh.paint_uniform_color([1.0, 0.1, 0.1])
    return mesh


def sanitize_cloud(cloud: np.ndarray) -> np.ndarray:
    """Remove invalid, excessively far, and overly dense points."""
    cloud = np.asarray(cloud, dtype=np.float64)

    valid = np.isfinite(cloud).all(axis=1)
    cloud = cloud[valid]

    # 保留車輛周圍較容易看清楚的區域
    mask = (
        (cloud[:, 0] > -20.0)
        & (cloud[:, 0] < 60.0)
        & (cloud[:, 1] > -35.0)
        & (cloud[:, 1] < 35.0)
        & (cloud[:, 2] > -5.0)
        & (cloud[:, 2] < 8.0)
    )
    cloud = cloud[mask]

    # 避免點雲過大導致 Open3D render 太慢
    max_points =  1_000_000 #100_000
    if len(cloud) > max_points:
        indices = np.random.choice(len(cloud), max_points, replace=False)
        cloud = cloud[indices]

    return cloud.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    lidar_dir = Path(f"{args.pred_dir}/lidar")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions = np.load(pred_dir / "predictions.npz")
    pred_xyz_all = predictions["pred_xyz"]
    t0_ns_all = predictions["t0_ns"]

    frame_count = len(pred_xyz_all)
    if args.max_frames > 0:
        frame_count = min(frame_count, args.max_frames)

    vis = o3d.visualization.Visualizer()

    # visible=False 可直接存圖；若要看即時畫面則加 --show
    vis.create_window(
        window_name="A-R1 trajectory on ROS LiDAR",
        width=1920,
        height=1080,
        visible=args.show,
    )

    render_option = vis.get_render_option()
    render_option.point_size = 2.0
    render_option.background_color = np.array([0.08, 0.08, 0.08])
    render_option.light_on = True

    for i in range(frame_count):
        t0_ns = int(t0_ns_all[i])

        lidar_file = lidar_dir / f"lidar_{i:04d}_{t0_ns}.npy"
        cloud_raw = np.load(lidar_file)
        cloud = sanitize_cloud(cloud_raw)

        traj_xyz = pred_xyz_all[i]

        print(f"\nFrame {i}")
        print("raw cloud shape:", cloud_raw.shape)
        print("filtered cloud shape:", cloud.shape)

        if len(cloud) == 0:
            print("WARNING: no valid LiDAR points after filtering. Skip this frame.")
            continue

        print("cloud min:", cloud.min(axis=0))
        print("cloud max:", cloud.max(axis=0))
        print("trajectory min:", traj_xyz.min(axis=0))
        print("trajectory max:", traj_xyz.max(axis=0))

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud)
        pcd.paint_uniform_color([0.65, 0.65, 0.65])

        trajectory = make_trajectory(traj_xyz, radius=0.25)

        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=2.0,
            origin=[0.0, 0.0, 0.0],
        )

        vis.clear_geometries()

        # 第一個 geometry 必須 reset bounding box=True
        vis.add_geometry(pcd, reset_bounding_box=True)
        vis.add_geometry(trajectory, reset_bounding_box=False)
        vis.add_geometry(axes, reset_bounding_box=False)

        # 讓 Open3D 先真正建立 scene，再設定 camera
        vis.poll_events()
        vis.update_renderer()

        ctr = vis.get_view_control()

        # 觀看車輛前方：X 向前、Y 向左、Z 向上
        ctr.set_lookat([15.0, 0.0, 0.0])
        ctr.set_front([-0.85, 0.0, 0.45])
        ctr.set_up([0.0, 0.0, 1.0])
        ctr.set_zoom(0.22)

        vis.poll_events()
        vis.update_renderer()

        output_path = out_dir / f"frame_{i:04d}_{t0_ns}.png"
        vis.capture_screen_image(str(output_path), do_render=True)

        print("saved:", output_path)

    vis.destroy_window()


if __name__ == "__main__":
    main()
