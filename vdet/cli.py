"""Command-line entry point.

Examples:
  python -m vdet.cli --source traffic.mp4
  python -m vdet.cli --source traffic.mp4 --classes car truck bus --line 0 0.6 1 0.6
  python -m vdet.cli --source 0 --show            # webcam
  python -m vdet.cli --config configs/default.yaml --source clip.mp4 --model yolo11s.pt --device 0
"""
from __future__ import annotations

import argparse
import json
import sys

from .config import PipelineConfig
from .pipeline import VideoPipeline


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Object detection + tracking + counting on video")
    p.add_argument("--source", required=True, help="video path, URL/RTSP stream, or webcam index")
    p.add_argument("--out", default="runs/latest", help="output directory")
    p.add_argument("--config", help="YAML config file (CLI flags override it)")
    p.add_argument("--model")
    p.add_argument("--conf", type=float)
    p.add_argument("--iou", type=float)
    p.add_argument("--imgsz", type=int)
    p.add_argument("--device")
    p.add_argument("--classes", nargs="+", help="class names to keep, e.g. person car")
    p.add_argument("--tracker", choices=["bytetrack.yaml", "botsort.yaml"])
    p.add_argument("--line", nargs=4, type=float, metavar=("X1", "Y1", "X2", "Y2"),
                   help="counting line in normalized 0-1 coords")
    p.add_argument("--no-line", action="store_true", help="disable line counting")
    p.add_argument("--stride", type=int)
    p.add_argument("--max-frames", type=int)
    p.add_argument("--show", action="store_true", default=None)
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    overrides = dict(model=a.model, conf=a.conf, iou=a.iou, imgsz=a.imgsz, device=a.device,
                     classes=a.classes, tracker=a.tracker,
                     line=tuple(a.line) if a.line else None,
                     stride=a.stride, max_frames=a.max_frames, show=a.show)
    if a.config:
        cfg = PipelineConfig.from_yaml(a.config, **overrides)
    else:
        cfg = PipelineConfig(**{k: v for k, v in overrides.items() if v is not None})
    if a.no_line:
        cfg.line = None

    source = int(a.source) if a.source.isdigit() else a.source

    def progress(i, n):
        if n:
            sys.stderr.write(f"\r{i}/{n} frames ({100 * i / n:5.1f}%)")
            sys.stderr.flush()

    summary = VideoPipeline(cfg).run(source, a.out, progress=progress)
    sys.stderr.write("\n")
    print(json.dumps({k: summary[k] for k in ("unique_objects", "line_crossings", "performance")}, indent=2))
    print(f"Outputs written to {a.out}/")


if __name__ == "__main__":
    main()
