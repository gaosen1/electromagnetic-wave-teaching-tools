#!/usr/bin/env python3
"""Render one of the three dispersion teaching videos."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from dispersion_demo import pcb_timeline, radar_timeline, timeline  # noqa: E402


DEMOS = {
    "fiber": (timeline.DURATION, timeline.render_frame, "系统色散动态演示_需求一.mp4"),
    "radar": (radar_timeline.DURATION, radar_timeline.render_frame, "系统色散动态演示_场景二_宽带雷达.mp4"),
    "pcb": (pcb_timeline.DURATION, pcb_timeline.render_frame, "系统色散动态演示_场景三_高速PCB.mp4"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", choices=DEMOS, default="fiber")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--frame", type=float, help="Render one timestamp as a PNG instead of a video")
    return parser.parse_args()


def render_video(args: argparse.Namespace, duration: float, renderer) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{args.width}x{args.height}",
        "-r",
        str(args.fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(args.output),
    ]
    frame_count = round(duration * args.fps)
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for frame_index in range(frame_count):
            seconds = frame_index / args.fps
            frame = renderer(args.width, args.height, seconds)
            process.stdin.write(frame.tobytes())
            if frame_index % args.fps == 0:
                print(f"rendered {frame_index // args.fps:02d}s / {duration:02.0f}s", flush=True)
    finally:
        process.stdin.close()
    return_code = process.wait()
    if return_code != 0:
        raise SystemExit(return_code)


def main() -> None:
    args = parse_args()
    default_duration, renderer, default_name = DEMOS[args.demo]
    duration = args.duration if args.duration is not None else default_duration
    if args.output is None:
        args.output = ROOT / "output" / default_name
    if args.frame is not None:
        output = args.output.with_suffix(".png")
        output.parent.mkdir(parents=True, exist_ok=True)
        renderer(args.width, args.height, args.frame).save(output)
        print(output)
        return
    render_video(args, duration, renderer)


if __name__ == "__main__":
    main()
