"""vdet: real-time object detection, tracking and analytics for video."""

from .config import PipelineConfig
from .analytics import LineCounter, TrackRegistry
from .pipeline import VideoPipeline

__all__ = ["PipelineConfig", "LineCounter", "TrackRegistry", "VideoPipeline"]
__version__ = "0.1.0"
