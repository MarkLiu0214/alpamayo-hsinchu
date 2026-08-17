#!/usr/bin/env python3
"""Print actual sensor frame IDs and all static TF edges from a ROS1 bag."""
import argparse
import rosbag

CAM_TOPICS = [
    "/lucid_cameras_x01/gige_100_fl_hdr/compressed",
    "/lucid_cameras_x00/gige_100_f_hdr/compressed",
    "/lucid_cameras_x01/gige_100_fr_hdr/compressed",
    "/lucid_cameras_x00/gige_30_f_hdr/compressed",
]
LIDAR_TOPIC = "/ouster/top_122219002200"
POSE_TOPIC = "/local_pose"

p = argparse.ArgumentParser(); p.add_argument("--bag", required=True); a = p.parse_args()
needed = set(CAM_TOPICS + [LIDAR_TOPIC, POSE_TOPIC, "/tf_static"])
seen = set()
with rosbag.Bag(a.bag) as bag:
    for topic, msg, _ in bag.read_messages(topics=list(needed)):
        if topic == "/tf_static":
            for transform in msg.transforms:
                h = transform.header
                tr = transform.transform.translation; q = transform.transform.rotation
                print(f"TF_STATIC parent={h.frame_id!r} child={transform.child_frame_id!r} "
                      f"t=({tr.x:.6f},{tr.y:.6f},{tr.z:.6f}) q_xyzw=({q.x:.8f},{q.y:.8f},{q.z:.8f},{q.w:.8f})")
            seen.add(topic)
        elif topic not in seen:
            header = getattr(msg, "header", None)
            print(f"TOPIC {topic}: type={msg._type}; header.frame_id={getattr(header, 'frame_id', None)!r}; "
                  f"stamp_ns={header.stamp.to_nsec() if header else None}")
            seen.add(topic)
        if seen >= needed:
            break
