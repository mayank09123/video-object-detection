"""Streamlit demo: upload a video, tune the detector, get an annotated video + analytics.

Run:  streamlit run app.py
"""
import json
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from vdet import PipelineConfig, VideoPipeline

COCO = ["person", "bicycle", "car", "motorcycle", "bus", "truck", "traffic light", "dog",
        "cat", "backpack", "handbag", "suitcase", "sports ball", "bottle", "cup", "chair",
        "laptop", "cell phone"]

st.set_page_config(page_title="Video Object Detection", layout="wide")
st.title("Video object detection & tracking")
st.caption("YOLO11 detection · ByteTrack/BoT-SORT multi-object tracking · line-crossing analytics")

with st.sidebar:
    st.header("Model")
    model = st.selectbox("Weights", ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"],
                         help="n = fastest, m = most accurate")
    conf = st.slider("Confidence threshold", 0.05, 0.95, 0.35, 0.05)
    imgsz = st.select_slider("Inference size", [320, 480, 640, 960], 640)
    device = st.text_input("Device", "cpu", help='"cpu", "0" for GPU, "mps" for Apple Silicon')
    classes = st.multiselect("Classes (empty = all 80)", COCO)
    st.header("Tracking")
    tracker = st.radio("Tracker", ["bytetrack.yaml", "botsort.yaml"], horizontal=True)
    st.header("Counting line (normalized)")
    use_line = st.checkbox("Enable line counting", True)
    c1, c2 = st.columns(2)
    y1 = c1.slider("Left y", 0.0, 1.0, 0.5, 0.01)
    y2 = c2.slider("Right y", 0.0, 1.0, 0.5, 0.01)
    st.header("Speed")
    stride = st.number_input("Process every Nth frame", 1, 10, 1)
    max_frames = st.number_input("Max frames (0 = all)", 0, 100000, 300)

upload = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])

if upload and st.button("Run detection", type="primary"):
    work = Path(tempfile.mkdtemp())
    src = work / upload.name
    src.write_bytes(upload.read())

    cfg = PipelineConfig(model=model, conf=conf, imgsz=imgsz, device=device,
                         classes=classes or None, tracker=tracker,
                         line=(0.0, y1, 1.0, y2) if use_line else None,
                         stride=int(stride), max_frames=int(max_frames) or None)
    bar = st.progress(0.0, text="Loading model…")

    def progress(i, n):
        if n:
            bar.progress(min(i / n, 1.0), text=f"Processing frame {i}/{n}")

    summary = VideoPipeline(cfg).run(str(src), work / "out", progress=progress)
    bar.empty()
    out = work / "out"

    left, right = st.columns([3, 2])
    with left:
        st.video(str(out / summary["outputs"]["video"]))
    with right:
        perf = summary["performance"]
        m1, m2, m3 = st.columns(3)
        m1.metric("Unique objects", sum(summary["unique_objects"].values()))
        m2.metric("Inference FPS", perf["inference_fps"])
        m3.metric("Mean latency", f'{perf["mean_inference_ms"]} ms')
        st.subheader("Objects by class")
        if summary["unique_objects"]:
            st.bar_chart(pd.Series(summary["unique_objects"], name="count"))
        if summary["line_crossings"]:
            st.subheader("Line crossings")
            st.dataframe(pd.DataFrame(summary["line_crossings"]).T)

    st.subheader("Tracks")
    tracks = pd.read_csv(out / "tracks.csv") if (out / "tracks.csv").stat().st_size > 1 else pd.DataFrame()
    st.dataframe(tracks, use_container_width=True)

    d1, d2, d3, d4 = st.columns(4)
    d1.download_button("Annotated video", (out / summary["outputs"]["video"]).read_bytes(), "annotated.mp4")
    d2.download_button("tracks.csv", (out / "tracks.csv").read_bytes(), "tracks.csv")
    d3.download_button("events.csv", (out / "events.csv").read_bytes(), "events.csv")
    d4.download_button("summary.json", json.dumps(summary, indent=2), "summary.json")
