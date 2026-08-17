#!/usr/bin/env python3
"""Create a composite MP4 from ROS-bag A-R1 inference outputs.

Expected inputs
---------------
<infer_dir>/predictions.npz
<infer_dir>/cot_0000_<t0_ns>.txt          (optional)
<lidar_vis_dir>/frame_0000_<t0_ns>.png
<cache_dir>/cameras/<camera_key>/<timestamp_ns>.jpg

The layout is:
  - Top: cross-left | front-wide | front-tele | cross-right
  - Bottom-left: LiDAR + predicted trajectory image
  - Bottom-right: timestamp, step number, predicted trajectory summary, optional CoT

It does NOT re-run inference or visualization. It only combines existing images.
"""
from __future__ import annotations

import argparse
import bisect
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Preserve the exact A-R1 camera ordering used by the ROS loader.
CAMERA_LAYOUT = [
    ("cross_left", "Cross Left"),
    ("front_wide", "Front Wide"),
    ("front_tele", "Front Tele"),
    ("cross_right", "Cross Right"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an MP4 combining ROS camera views, LiDAR overlay, and A-R1 inference text."
    )
    parser.add_argument("--infer-dir", required=True, help="Directory containing predictions.npz and cot_*.txt")
    parser.add_argument("--cache", required=True, help="ROS bag cache created by extract_rosbag_ar1.py")
    parser.add_argument("--lidar-vis-dir", required=True, help="Directory containing frame_XXXX_<t0_ns>.png")
    parser.add_argument("--out", required=True, help="Output .mp4 filename")
    parser.add_argument("--fps", type=float, default=1.0, help="Output video FPS. Default: 1")
    parser.add_argument(
        "--seconds-per-frame",
        type=float,
        default=1.0,
        help="How long each inference frame remains on screen. Default: 1 second",
    )
    parser.add_argument("--max-frames", type=int, default=0, help="0 means use every saved prediction")
    parser.add_argument(
        "--camera-tolerance-ms",
        type=float,
        default=120.0,
        help="Warn when nearest camera image differs from t0 by more than this tolerance. Default: 120 ms",
    )
    parser.add_argument(
        "--no-cot",
        action="store_true",
        help="Do not display Chain-of-Causation text even if cot_*.txt exists",
    )
    parser.add_argument(
        "--keep-aspect",
        action="store_true",
        help="Pad images instead of cropping when fitting them into panels",
    )
    return parser.parse_args()


def find_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend(
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            ]
        )
    candidates.extend(
        [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]
    )
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def indexed_camera_files(cache_dir: Path) -> Dict[str, Tuple[List[int], List[Path]]]:
    """Build a sorted timestamp index for each cached camera directory."""
    output: Dict[str, Tuple[List[int], List[Path]]] = {}
    for key, _ in CAMERA_LAYOUT:
        directory = cache_dir / "cameras" / key
        records = []
        for path in directory.glob("*.jpg"):
            try:
                records.append((int(path.stem), path))
            except ValueError:
                continue
        records.sort(key=lambda item: item[0])
        if not records:
            raise FileNotFoundError(f"No JPG images found in {directory}")
        timestamps, paths = zip(*records)
        output[key] = (list(timestamps), list(paths))
    return output


def nearest_camera_path(
    index: Tuple[List[int], List[Path]], target_ns: int
) -> Tuple[Path, int]:
    timestamps, paths = index
    pos = bisect.bisect_left(timestamps, target_ns)
    candidates = []
    if pos < len(timestamps):
        candidates.append(pos)
    if pos > 0:
        candidates.append(pos - 1)
    best = min(candidates, key=lambda k: abs(timestamps[k] - target_ns))
    return paths[best], timestamps[best]


def fit_image(image: Image.Image, size: Tuple[int, int], keep_aspect: bool) -> Image.Image:
    """Fit image to a panel; either crop-fill or letterbox."""
    target_w, target_h = size
    image = image.convert("RGB")
    src_w, src_h = image.size

    if keep_aspect:
        scale = min(target_w / src_w, target_h / src_h)
        resized = image.resize((max(1, round(src_w * scale)), max(1, round(src_h * scale))), Image.Resampling.LANCZOS)
        panel = Image.new("RGB", size, (20, 20, 20))
        x = (target_w - resized.width) // 2
        y = (target_h - resized.height) // 2
        panel.paste(resized, (x, y))
        return panel

    scale = max(target_w / src_w, target_h / src_h)
    resized = image.resize((max(1, round(src_w * scale)), max(1, round(src_h * scale))), Image.Resampling.LANCZOS)
    left = max(0, (resized.width - target_w) // 2)
    top = max(0, (resized.height - target_h) // 2)
    return resized.crop((left, top, left + target_w, top + target_h))


def crop_lidar_render(image: Image.Image) -> Image.Image:
    """Remove narrow Open3D window borders when they are present."""
    image = image.convert("RGB")
    w, h = image.size
    # Small crop only. It avoids accidentally cutting useful road/trajectory content.
    margin_x = int(w * 0.01)
    margin_y = int(h * 0.015)
    return image.crop((margin_x, margin_y, w - margin_x, h - margin_y))


def read_cot(infer_dir: Path, step: int, t0_ns: int) -> str:
    cot_dir = Path(f"{infer_dir}/cot")
    path = cot_dir / f"cot_{step:04d}_{t0_ns}.txt"
    if not path.exists():
        return "No Chain-of-Causation text was saved for this frame."
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    # Output can look like nested Python lists; this makes it much more readable.
    text = text.replace("[['", "").replace("']]", "").replace("\\n", " ")
    text = " ".join(text.split())
    return text or "No Chain-of-Causation text was saved for this frame."


def draw_text_box(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[int, int, int, int],
    title: str,
    body: str,
    title_font: ImageFont.ImageFont,
    body_font: ImageFont.ImageFont,
) -> None:
    x0, y0, x1, y1 = xy
    draw.rounded_rectangle(xy, radius=16, fill=(28, 31, 37), outline=(85, 92, 105), width=2)
    draw.text((x0 + 24, y0 + 18), title, fill=(235, 235, 235), font=title_font)

    text_y = y0 + 66
    max_width_px = (x1 - x0) - 48
    lines: List[str] = []
    # Pixel-aware wrapping for proportional fonts.
    for paragraph in body.splitlines() or [body]:
        words = paragraph.split()
        line = ""
        for word in words:
            candidate = word if not line else f"{line} {word}"
            if draw.textlength(candidate, font=body_font) <= max_width_px:
                line = candidate
            else:
                if line:
                    lines.append(line)
                line = word
        if line:
            lines.append(line)

    line_height = max(22, int(getattr(body_font, "size", 20) * 1.35))
    max_lines = max(1, (y1 - text_y - 18) // line_height)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        if lines:
            lines[-1] = textwrap.shorten(lines[-1], width=max(10, len(lines[-1]) - 4), placeholder=" …")

    for line in lines:
        draw.text((x0 + 24, text_y), line, fill=(218, 220, 224), font=body_font)
        text_y += line_height


def lidar_path_for(lidar_vis_dir: Path, step: int, t0_ns: int) -> Path:
    exact = lidar_vis_dir / f"frame_{step:04d}_{t0_ns}.png"
    if exact.exists():
        return exact
    candidates = sorted(lidar_vis_dir.glob(f"frame_{step:04d}_*.png"))
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(
        f"Cannot locate LiDAR visualization for step={step}, t0_ns={t0_ns}. Expected {exact}"
    )


def create_composite_frame(
    *,
    step: int,
    t0_ns: int,
    traj_xyz: np.ndarray,
    camera_index: Dict[str, Tuple[List[int], List[Path]]],
    infer_dir: Path,
    lidar_vis_dir: Path,
    camera_tolerance_ns: int,
    show_cot: bool,
    keep_aspect: bool,
) -> np.ndarray:
    # 1920x1080 works well with typical Open3D output and can be encoded quickly.
    canvas_w, canvas_h = 1920, 1080
    top_h = 270
    gap = 12
    camera_w = (canvas_w - gap * 3) // 4
    camera_h = top_h
    lower_y = top_h + gap
    lower_h = canvas_h - lower_y
    lidar_w = 1240
    text_x = lidar_w + gap
    text_w = canvas_w - text_x

    frame = Image.new("RGB", (canvas_w, canvas_h), (15, 17, 21))
    draw = ImageDraw.Draw(frame)
    label_font = find_font(24, bold=True)
    title_font = find_font(28, bold=True)
    body_font = find_font(20)
    small_font = find_font(18)

    # Four synchronized camera images.
    for index, (camera_key, camera_name) in enumerate(CAMERA_LAYOUT):
        path, image_t_ns = nearest_camera_path(camera_index[camera_key], t0_ns)
        with Image.open(path) as im:
            panel = fit_image(im, (camera_w, camera_h), keep_aspect)
        x = index * (camera_w + gap)
        frame.paste(panel, (x, 0))
        draw.rectangle((x, 0, x + camera_w, 34), fill=(0, 0, 0))
        delta_ms = (image_t_ns - t0_ns) / 1e6
        suffix = f"  Δt={delta_ms:+.1f} ms"
        label_color = (255, 198, 80) if abs(image_t_ns - t0_ns) > camera_tolerance_ns else (245, 245, 245)
        draw.text((10 + x, 5), camera_name + suffix, fill=label_color, font=small_font)

    # Existing LiDAR + trajectory render.
    lidar_path = lidar_path_for(lidar_vis_dir, step, t0_ns)
    with Image.open(lidar_path) as im:
        lidar_panel = fit_image(crop_lidar_render(im), (lidar_w, lower_h), keep_aspect)
    frame.paste(lidar_panel, (0, lower_y))
    draw.rectangle((0, lower_y, lidar_w, lower_y + 40), fill=(0, 0, 0))
    draw.text((16, lower_y + 8), "Top LiDAR + Alpamayo-R1 predicted trajectory", fill=(245, 245, 245), font=label_font)

    # Right column: metadata and CoT.
    x0, x1 = text_x, canvas_w - 12
    meta_height = 260    # 215
    xy_meta = (x0, lower_y, x1, lower_y + meta_height)
    horizon_s = len(traj_xyz) * 0.1
    final_xy = traj_xyz[-1, :2]
    final_dist = float(np.linalg.norm(final_xy))
    timestamp_s = t0_ns / 1e9
    meta = (
        f"Inference step: {step}\n"
        f"t0: {t0_ns} ns ({timestamp_s:.3f} s)\n"
        f"Prediction: {len(traj_xyz)} points / {horizon_s:.1f} s horizon\n"
        f"Final predicted position: x={final_xy[0]:.2f} m, y={final_xy[1]:.2f} m\n"
        f"Final planar distance: {final_dist:.2f} m"
    )
    draw_text_box(draw, xy_meta, "ROS bag A-R1 inference", meta, title_font, body_font)

    cot_title = "Chain-of-Causation" if show_cot else "Notes"
    cot_body = read_cot(infer_dir, step, t0_ns) if show_cot else "CoT display disabled by --no-cot."
    draw_text_box(
        draw,
        (x0, lower_y + meta_height + gap, x1, canvas_h - 12),
        cot_title,
        cot_body,
        title_font,
        body_font,
    )

    # PIL RGB -> OpenCV BGR for VideoWriter.
    return cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be > 0")
    if args.seconds_per_frame <= 0:
        raise ValueError("--seconds-per-frame must be > 0")

    infer_dir = Path(args.infer_dir)
    cache_dir = Path(args.cache)
    lidar_vis_dir = Path(args.lidar_vis_dir)
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prediction_file = infer_dir / "predictions.npz"
    if not prediction_file.exists():
        raise FileNotFoundError(f"Missing {prediction_file}")

    saved = np.load(prediction_file)
    t0_ns = np.asarray(saved["t0_ns"], dtype=np.int64)
    pred_xyz = np.asarray(saved["pred_xyz"], dtype=np.float32)
    if len(t0_ns) != len(pred_xyz):
        raise RuntimeError("predictions.npz has mismatched t0_ns and pred_xyz lengths")

    frame_count = len(t0_ns)
    if args.max_frames > 0:
        frame_count = min(frame_count, args.max_frames)
    if frame_count == 0:
        raise RuntimeError("No prediction frames found")

    camera_index = indexed_camera_files(cache_dir)
    camera_tolerance_ns = int(args.camera_tolerance_ms * 1e6)
    repeats = max(1, round(args.seconds_per_frame * args.fps))

    width, height = 1920, 1080
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, args.fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(
            "OpenCV could not create the MP4. Install ffmpeg/GStreamer support or use a writable output path."
        )

    try:
        for step in range(frame_count):
            current_t0_ns = int(t0_ns[step])
            print(f"[{step + 1}/{frame_count}] composing t0_ns={current_t0_ns}")
            composite = create_composite_frame(
                step=step,
                t0_ns=current_t0_ns,
                traj_xyz=pred_xyz[step],
                camera_index=camera_index,
                infer_dir=infer_dir,
                lidar_vis_dir=lidar_vis_dir,
                camera_tolerance_ns=camera_tolerance_ns,
                show_cot=not args.no_cot,
                keep_aspect=args.keep_aspect,
            )
            for _ in range(repeats):
                writer.write(composite)
    finally:
        writer.release()

    duration = frame_count * repeats / args.fps
    print(f"Saved MP4: {output_path}")
    print(f"Frames in video: {frame_count * repeats}; duration: {duration:.2f} s; fps: {args.fps}")


if __name__ == "__main__":
    main()
