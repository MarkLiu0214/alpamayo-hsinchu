#!/usr/bin/env python3
"""Run A-R1 on a ROS bag cache and save one prediction per inference time."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper
from load_rosbag_with_lidar import RosbagAR1Cache, load_rosbag_with_lidar


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--start-offset-s", type=float, default=3.0,
                   help="Seconds after first pose used as first t0; must be >=1.5")
    p.add_argument("--stride-s", type=float, default=1.0)
    p.add_argument("--num-steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    cache = RosbagAR1Cache(args.cache)
    valid_start, valid_end = cache.valid_t0_range_ns()
    start_ns = max(valid_start, int(cache.pose_t[0] + args.start_offset_s * 1e9))
    stride_ns = int(args.stride_s * 1e9)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    out_pred = Path(f"{args.out}/pred"); out_pred.mkdir(parents=True, exist_ok=True)
    out_lidar = Path(f"{args.out}/lidar"); out_lidar.mkdir(parents=True, exist_ok=True)
    out_cot = Path(f"{args.out}/cot"); out_cot.mkdir(parents=True, exist_ok=True)
    
    model = AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B", dtype=torch.bfloat16).to("cuda").eval()
    processor = helper.get_processor(model.tokenizer)
    saved_t0, saved_pred, saved_lidar_t = [], [], []

    for step in range(args.num_steps):
        t0_ns = start_ns + step * stride_ns
        if t0_ns > valid_end:
            print(f"Stop: t0={t0_ns} exceeds pose range")
            break
        data = load_rosbag_with_lidar(cache, t0_ns)
        messages = helper.create_message(data["image_frames"].flatten(0, 1))
        inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=False,
                                               continue_final_message=True, return_dict=True, return_tensors="pt")
        model_inputs = helper.to_device({
            "tokenized_data": inputs,
            "ego_history_xyz": data["ego_history_xyz"],
            "ego_history_rot": data["ego_history_rot"],
        }, "cuda")
        torch.cuda.manual_seed_all(args.seed)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs, top_p=0.98, temperature=0.6, num_traj_samples=1,
                max_generation_length=256, return_extra=True,
            )
        # Typical shape is [B, sample, rollout, 64, 3]. Store only the 64 ego-frame XYZ points.
        xyz = pred_xyz.detach().float().cpu().numpy()[0, 0, 0, :, :3]
        np.save(out_pred / f"pred_{step:04d}_{t0_ns}.npy", xyz)
        np.save(out_lidar / f"lidar_{step:04d}_{t0_ns}.npy", data["point_cloud"])
        saved_t0.append(t0_ns); saved_pred.append(xyz); saved_lidar_t.append(data["lidar_t_ns"])
        (out_cot / f"cot_{step:04d}_{t0_ns}.txt").write_text(str(extra["cot"][0]), encoding="utf-8")
        print(f"step={step}: saved 64 points; lidar={data['lidar_t_ns']}; cot={str(extra['cot'][0])[:100]}")

    np.savez_compressed(out / "predictions.npz", t0_ns=np.asarray(saved_t0),
                        lidar_t_ns=np.asarray(saved_lidar_t), pred_xyz=np.asarray(saved_pred))

if __name__ == "__main__":
    main()
