"""Tracking analytics that don't depend on the model: line crossing and per-track stats.

Kept free of torch/ultralytics imports so it is fast to unit-test.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field


Point = tuple[float, float]


def _side(p: Point, a: Point, b: Point) -> float:
    """Signed area: >0 if p is left of a->b, <0 if right, 0 on the line."""
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def _segments_intersect(p1: Point, p2: Point, a: Point, b: Point) -> bool:
    d1, d2 = _side(p1, a, b), _side(p2, a, b)
    d3, d4 = _side(a, p1, p2), _side(b, p1, p2)
    return (d1 * d2 < 0) and (d3 * d4 < 0)


@dataclass
class CrossingEvent:
    frame: int
    timestamp_s: float
    track_id: int
    class_name: str
    direction: str  # "in" (left->right of the line vector) or "out"


class LineCounter:
    """Counts tracks whose centroid path crosses a line segment.

    Each track is counted at most once per direction, which avoids double counts
    when a box jitters on top of the line.
    """

    def __init__(self, a: Point, b: Point):
        self.a, self.b = a, b
        self._last: dict[int, Point] = {}
        self._counted: set[tuple[int, str]] = set()
        self.counts: dict[str, dict[str, int]] = defaultdict(lambda: {"in": 0, "out": 0})
        self.events: list[CrossingEvent] = []

    def update(self, track_id: int, centroid: Point, class_name: str,
               frame: int = 0, timestamp_s: float = 0.0) -> CrossingEvent | None:
        prev = self._last.get(track_id)
        self._last[track_id] = centroid
        if prev is None or not _segments_intersect(prev, centroid, self.a, self.b):
            return None
        direction = "in" if _side(centroid, self.a, self.b) > 0 else "out"
        key = (track_id, direction)
        if key in self._counted:
            return None
        self._counted.add(key)
        self.counts[class_name][direction] += 1
        ev = CrossingEvent(frame, round(timestamp_s, 3), track_id, class_name, direction)
        self.events.append(ev)
        return ev

    @property
    def total_in(self) -> int:
        return sum(c["in"] for c in self.counts.values())

    @property
    def total_out(self) -> int:
        return sum(c["out"] for c in self.counts.values())


@dataclass
class TrackInfo:
    track_id: int
    class_votes: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    first_frame: int = 0
    last_frame: int = 0
    n_detections: int = 0
    conf_sum: float = 0.0
    path_px: float = 0.0
    trail: deque = field(default_factory=lambda: deque(maxlen=30))

    @property
    def class_name(self) -> str:
        # Majority vote smooths out frame-level class flicker (e.g. car <-> truck)
        return max(self.class_votes.items(), key=lambda kv: kv[1])[0]

    @property
    def mean_conf(self) -> float:
        return self.conf_sum / max(self.n_detections, 1)


class TrackRegistry:
    """Accumulates per-track lifetime statistics across the video."""

    def __init__(self, trail_length: int = 30):
        self.trail_length = trail_length
        self.tracks: dict[int, TrackInfo] = {}

    def update(self, track_id: int, class_name: str, conf: float,
               centroid: Point, frame: int) -> TrackInfo:
        t = self.tracks.get(track_id)
        if t is None:
            t = TrackInfo(track_id, first_frame=frame, trail=deque(maxlen=self.trail_length))
            self.tracks[track_id] = t
        if t.trail:
            px, py = t.trail[-1]
            t.path_px += ((centroid[0] - px) ** 2 + (centroid[1] - py) ** 2) ** 0.5
        t.trail.append(centroid)
        t.class_votes[class_name] += 1
        t.last_frame = frame
        t.n_detections += 1
        t.conf_sum += conf
        return t

    def summary(self, fps: float, min_detections: int = 3) -> list[dict]:
        out = []
        for t in self.tracks.values():
            if t.n_detections < min_detections:  # drop flicker / one-off false positives
                continue
            out.append({
                "track_id": t.track_id,
                "class": t.class_name,
                "first_seen_s": round(t.first_frame / fps, 2),
                "last_seen_s": round(t.last_frame / fps, 2),
                "duration_s": round((t.last_frame - t.first_frame) / fps, 2),
                "detections": t.n_detections,
                "mean_conf": round(t.mean_conf, 3),
                "path_px": round(t.path_px, 1),
            })
        return sorted(out, key=lambda r: r["track_id"])

    def unique_counts(self, min_detections: int = 3) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for t in self.tracks.values():
            if t.n_detections >= min_detections:
                counts[t.class_name] += 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))
