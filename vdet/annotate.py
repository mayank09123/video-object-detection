from __future__ import annotations

import colorsys

import cv2
import numpy as np


def color_for(track_id: int) -> tuple[int, int, int]:
    """Stable, well-separated BGR color per track id (golden-ratio hue stepping)."""
    h = (track_id * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.75, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def draw_label(img, text, org, color, scale=0.5, thickness=1):
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = org
    y = max(y, h + 4)
    cv2.rectangle(img, (x, y - h - base - 4), (x + w + 4, y), color, -1)
    cv2.putText(img, text, (x + 2, y - base - 2), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), thickness, cv2.LINE_AA)


def draw_track(img, box, track_id, class_name, conf, trail):
    color = color_for(track_id)
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    draw_label(img, f"#{track_id} {class_name} {conf:.2f}", (x1, y1), color)
    if len(trail) > 1:
        pts = np.array(trail, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], False, color, 2, cv2.LINE_AA)


def draw_line(img, a, b, color=(0, 255, 255)):
    cv2.line(img, tuple(map(int, a)), tuple(map(int, b)), color, 2, cv2.LINE_AA)


def draw_hud(img, lines: list[str]):
    """Semi-transparent stats panel in the top-left corner."""
    if not lines:
        return
    pad, lh = 8, 20
    w = max(cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] for s in lines) + 2 * pad
    h = lh * len(lines) + pad
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    for i, s in enumerate(lines):
        cv2.putText(img, s, (pad, pad + 12 + i * lh), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
