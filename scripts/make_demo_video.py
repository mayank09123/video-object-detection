"""Generate a small synthetic test clip so the pipeline can be run without downloading footage.

It slides a camera window over Ultralytics' sample `bus.jpg`, so the people and bus
move downward through the frame and cross the default horizontal counting line.

For real evaluation use actual footage, e.g. free clips from pexels.com/videos
(search "traffic", "pedestrians", "intersection") or the MOT17 benchmark.
"""
import argparse
import os

import cv2
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/demo.mp4")
    p.add_argument("--seconds", type=float, default=6)
    p.add_argument("--fps", type=int, default=25)
    a = p.parse_args()

    import ultralytics
    img = cv2.imread(os.path.join(os.path.dirname(ultralytics.__file__), "assets", "bus.jpg"))
    H, W = img.shape[:2]                 # 1080 x 810
    win_h = H // 2
    n = int(a.seconds * a.fps)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (W, win_h))
    for i in range(n):
        # window moves bottom -> top, so scene content appears to move downward
        y = int((H - win_h) * (1 - i / max(n - 1, 1)))
        frame = img[y:y + win_h].copy()
        frame = cv2.add(frame, np.random.randint(0, 6, frame.shape, dtype=np.uint8))  # sensor noise
        vw.write(frame)
    vw.release()
    print(f"wrote {a.out} ({n} frames, {W}x{win_h} @ {a.fps}fps)")


if __name__ == "__main__":
    main()
