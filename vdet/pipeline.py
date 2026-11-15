from __future__ import annotations

import csv
import json
import shutil
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from .analytics import LineCounter, TrackRegistry
from .annotate import draw_hud, draw_line, draw_track
from .config import PipelineConfig

ProgressFn = Callable[[int, int], None]


class VideoPipeline:
    """Detect -> track -> analyze -> annotate, frame by frame.

    Produces in `out_dir`:
      annotated.mp4   video with boxes, IDs, trails, counting line and HUD
      events.csv      one row per line-crossing event
      tracks.csv      one row per unique tracked object (lifetime stats)
      summary.json    run config, unique object counts, crossings, throughput
    """

    def __init__(self, cfg: PipelineConfig):
        from ultralytics import YOLO  # imported lazily so analytics stays torch-free

        self.cfg = cfg
        self.model = YOLO(cfg.model)
        self.names: dict[int, str] = self.model.names
        self.class_ids = self._resolve_classes(cfg.classes)

    def _resolve_classes(self, classes: Optional[list[str]]) -> Optional[list[int]]:
        if not classes:
            return None
        lookup = {v.lower(): k for k, v in self.names.items()}
        unknown = [c for c in classes if c.lower() not in lookup]
        if unknown:
            raise ValueError(f"Unknown classes {unknown}. Model knows: {sorted(lookup)}")
        return [lookup[c.lower()] for c in classes]

    def run(self, source: str | int, out_dir: str | Path,
            progress: Optional[ProgressFn] = None) -> dict:
        cfg = self.cfg
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video source: {source}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if cfg.max_frames:
            total = min(total, cfg.max_frames) if total else cfg.max_frames

        raw_path = out_dir / "annotated_raw.mp4"
        out_fps = fps / cfg.stride
        writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (w, h))

        line_px = None
        counter = None
        if cfg.line:
            x1, y1, x2, y2 = cfg.line
            line_px = ((x1 * w, y1 * h), (x2 * w, y2 * h))
            counter = LineCounter(*line_px)
        registry = TrackRegistry(cfg.trail_length)

        frame_idx, processed = -1, 0
        infer_times: list[float] = []
        t_start = time.perf_counter()

        # Reset tracker state in case this model instance was used before
        if getattr(self.model, "predictor", None) is not None:
            self.model.predictor = None

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if cfg.max_frames and frame_idx >= cfg.max_frames:
                break
            if frame_idx % cfg.stride:
                continue

            t0 = time.perf_counter()
            res = self.model.track(
                frame, persist=True, tracker=cfg.tracker, conf=cfg.conf, iou=cfg.iou,
                imgsz=cfg.imgsz, classes=self.class_ids, device=cfg.device, verbose=False,
            )[0]
            infer_times.append(time.perf_counter() - t0)

            ts = frame_idx / fps
            active = 0
            boxes = res.boxes
            if boxes is not None and boxes.id is not None:
                xyxy = boxes.xyxy.cpu().numpy()
                ids = boxes.id.int().cpu().numpy()
                cls = boxes.cls.int().cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                for box, tid, c, cf in zip(xyxy, ids, cls, confs):
                    tid, name = int(tid), self.names[int(c)]
                    # Bottom-center is a better ground-contact point than box center for counting
                    cx, cy = float((box[0] + box[2]) / 2), float(box[3])
                    info = registry.update(tid, name, float(cf), (cx, cy), frame_idx)
                    if counter:
                        counter.update(tid, (cx, cy), info.class_name, frame_idx, ts)
                    draw_track(frame, box, tid, info.class_name, float(cf), list(info.trail))
                    active += 1

            if line_px:
                draw_line(frame, *line_px)
            live_fps = 1.0 / np.mean(infer_times[-20:])
            hud = [f"frame {frame_idx}  t={ts:6.2f}s", f"active tracks: {active}",
                   f"unique objects: {len(registry.tracks)}", f"inference: {live_fps:5.1f} FPS"]
            if counter:
                hud.append(f"line  in: {counter.total_in}  out: {counter.total_out}")
            draw_hud(frame, hud)

            writer.write(frame)
            processed += 1
            if cfg.show:
                cv2.imshow("vdet", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if progress:
                progress(frame_idx + 1, total)

        cap.release()
        writer.release()
        if cfg.show:
            cv2.destroyAllWindows()

        video_path = self._to_h264(raw_path, out_dir / "annotated.mp4")
        wall = time.perf_counter() - t_start

        tracks = registry.summary(fps)
        self._write_csv(out_dir / "tracks.csv", tracks)
        events = [asdict(e) for e in counter.events] if counter else []
        self._write_csv(out_dir / "events.csv", events,
                        fieldnames=["frame", "timestamp_s", "track_id", "class_name", "direction"])

        # Drop the first call: it includes model warmup / graph setup and skews latency stats
        stats_t = infer_times[1:] if len(infer_times) > 1 else infer_times
        summary = {
            "source": str(source),
            "video": {"width": w, "height": h, "fps": round(fps, 2), "frames_processed": processed},
            "config": cfg.to_dict(),
            "unique_objects": registry.unique_counts(),
            "line_crossings": ({k: dict(v) for k, v in counter.counts.items()} if counter else None),
            "performance": {
                "mean_inference_ms": round(1000 * float(np.mean(stats_t)), 1) if stats_t else None,
                "p95_inference_ms": round(1000 * float(np.percentile(stats_t, 95)), 1) if stats_t else None,
                "inference_fps": round(1.0 / float(np.mean(stats_t)), 1) if stats_t else None,
                "end_to_end_fps": round(processed / wall, 1) if wall else None,
                "wall_time_s": round(wall, 2),
            },
            "outputs": {"video": video_path.name, "tracks": "tracks.csv", "events": "events.csv"},
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        return summary

    @staticmethod
    def _to_h264(src: Path, dst: Path) -> Path:
        """OpenCV writes mp4v, which browsers won't play. Re-encode with ffmpeg if available."""
        if shutil.which("ffmpeg"):
            r = subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-c:v", "libx264",
                 "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)],
                capture_output=True,
            )
            if r.returncode == 0:
                src.unlink(missing_ok=True)
                return dst
        src.replace(dst)
        return dst

    @staticmethod
    def _write_csv(path: Path, rows: list[dict], fieldnames: Optional[list[str]] = None):
        fieldnames = fieldnames or (list(rows[0].keys()) if rows else [])
        with path.open("w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=fieldnames)
            wr.writeheader()
            wr.writerows(rows)
