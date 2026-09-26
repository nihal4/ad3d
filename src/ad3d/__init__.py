"""ad3d - a clean, Kaggle-ready 3D point-cloud anomaly detection workbench.

Baseline: Simple3D-lite (multi-scale FPFH + LFSA + prototype memory bank),
evaluated on Anomaly-ShapeNet and Real3D-AD with the official protocols.
"""
from .datasets import REAL3D_CLASSES, SHAPENET_CLASSES, get_classes  # noqa: F401
from .method import Config, Simple3DLite, run_benchmark  # noqa: F401
