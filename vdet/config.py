from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class PipelineConfig:
    # Model
    model: str = "yolo11n.pt"            # any Ultralytics detection model (yolo11n/s/m, yolov8*, custom .pt)
    conf: float = 0.35                    # detection confidence threshold
    iou: float = 0.5                      # NMS IoU threshold
    imgsz: int = 640                      # inference resolution
    device: str = "cpu"                   # "cpu", "0" (first GPU), "mps"
    classes: Optional[list[str]] = None   # filter by class name, e.g. ["person", "car"]; None = all 80 COCO classes

    # Tracking
    tracker: str = "bytetrack.yaml"       # "bytetrack.yaml" or "botsort.yaml"
    trail_length: int = 30                # frames of trajectory history drawn per track

    # Analytics: counting line in *normalized* coords (0-1), so it works at any resolution
    line: Optional[tuple[float, float, float, float]] = (0.0, 0.5, 1.0, 0.5)

    # I/O
    max_frames: Optional[int] = None      # stop early (handy for quick tests)
    stride: int = 1                       # process every Nth frame (speed vs. accuracy)
    show: bool = False                    # live preview window (needs a display)

    extra: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides) -> "PipelineConfig":
        data = yaml.safe_load(Path(path).read_text()) or {}
        data.update({k: v for k, v in overrides.items() if v is not None})
        if data.get("line") is not None:
            data["line"] = tuple(data["line"])
        return cls(**data)

    def to_dict(self) -> dict:
        return asdict(self)
