"""Unified dataset loading for Anomaly-ShapeNet and Real3D-AD.

Expected on-disk layouts (after extraction):

Anomaly-ShapeNet  (root = the `pcd` directory):
    <root>/<category>/train/*.pcd              4 normal prototypes
    <root>/<category>/test/*.pcd               normal + anomalous (see below)
    <root>/<category>/GT/*.txt                 anomalous samples: "x,y,z,flag"
                                               (the txt itself is the input cloud,
                                                mirroring the official benchmark code)
    A test .pcd is anomalous iff GT/<stem>.txt exists (its twin in test/ is
    ignored); every other test .pcd is a normal sample.

Real3D-AD  (root = the directory containing the 12 category folders):
    <root>/<category>/train/*.pcd              4 normal prototypes
    <root>/<category>/test/*.pcd               'good' in name -> normal (label 0)
    <root>/<category>/gt/<stem>.txt            anomalous: "x y z flag" (space separated)
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np
import open3d as o3d
import pandas as pd


# --------------------------------------------------------------------------- #
# Class lists (official benchmark order)
# --------------------------------------------------------------------------- #
SHAPENET_CLASSES = [
    "ashtray0", "bag0", "bottle0", "bottle1", "bottle3",
    "bowl0", "bowl1", "bowl2", "bowl3", "bowl4", "bowl5",
    "bucket0", "bucket1",
    "cap0", "cap3", "cap4", "cap5",
    "cup0", "cup1",
    "eraser0",
    "headset0", "headset1",
    "helmet0", "helmet1", "helmet2", "helmet3",
    "jar0",
    "microphone0",
    "shelf0",
    "tap0", "tap1",
    "vase0", "vase1", "vase2", "vase3", "vase4", "vase5", "vase7", "vase8", "vase9",
]

REAL3D_CLASSES = [
    "airplane", "candybar", "car", "chicken", "diamond", "duck",
    "fish", "gemstone", "seahorse", "shell", "starfish", "toffees",
]


def get_classes(dataset: str) -> list[str]:
    if dataset == "shapenet":
        return list(SHAPENET_CLASSES)
    if dataset == "real3d":
        return list(REAL3D_CLASSES)
    raise ValueError(f"Unknown dataset '{dataset}' (use 'shapenet' or 'real3d')")


# --------------------------------------------------------------------------- #
# Sample container
# --------------------------------------------------------------------------- #
@dataclass
class TestSample:
    points: np.ndarray   # (n, 3) float32
    gt: np.ndarray       # (n,) float32, 1 = anomalous point
    label: int           # 0 = normal object, 1 = anomalous object
    path: str


# --------------------------------------------------------------------------- #
# Low-level readers
# --------------------------------------------------------------------------- #
def read_pcd(path: str) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points, dtype=np.float32)
    if pts.shape[0] == 0:
        raise ValueError(f"Empty point cloud: {path}")
    return pts


def read_txt_cloud(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read a 4-column ground-truth txt (x y z flag).

    Handles both space and comma delimiters (Anomaly-ShapeNet uses ',',
    Real3D-AD uses spaces).
    """
    with open(path, "r") as f:
        first = f.readline()
    delim = "," if "," in first else None
    arr = np.loadtxt(path, dtype=np.float32, delimiter=delim)
    if arr.ndim == 1:
        arr = arr[None, :]
    points = arr[:, :3].astype(np.float32)
    gt = (arr[:, 3] > 0.5).astype(np.float32) if arr.shape[1] > 3 else np.zeros(len(points), np.float32)
    return points, gt


# --------------------------------------------------------------------------- #
# Dataset loaders
# --------------------------------------------------------------------------- #
def load_train(root: str, dataset: str, cls: str) -> list[np.ndarray]:
    """Return the list of raw training (normal) point clouds for one category."""
    if dataset == "shapenet":
        paths = sorted(glob.glob(os.path.join(root, cls, "train", "*.pcd")))
    elif dataset == "real3d":
        paths = sorted(glob.glob(os.path.join(root, cls, "train", "*.pcd")))
    else:
        raise ValueError(dataset)
    if not paths:
        raise FileNotFoundError(f"No training .pcd found under {os.path.join(root, cls, 'train')}")
    return [read_pcd(p) for p in paths]


def load_test(root: str, dataset: str, cls: str) -> list[TestSample]:
    """Return all test samples for one category (normal + anomalous)."""
    samples: list[TestSample] = []

    if dataset == "shapenet":
        test_dir = os.path.join(root, cls, "test")
        gt_dir = os.path.join(root, cls, "GT")
        # Normal test files carry 'positive' in the name (official convention).
        normal_paths = sorted(p for p in glob.glob(os.path.join(test_dir, "*.pcd"))
                              if "positive" in os.path.basename(p))
        for p in normal_paths:
            pts = read_pcd(p)
            samples.append(TestSample(pts, np.zeros(len(pts), np.float32), 0, p))
        # Anomalous samples: input cloud AND mask come from GT txt.
        for p in sorted(glob.glob(os.path.join(gt_dir, "*.txt"))):
            pts, gt = read_txt_cloud(p)
            samples.append(TestSample(pts, gt, 1, p))
        if not normal_paths:
            found = [os.path.basename(p) for p in
                     sorted(glob.glob(os.path.join(test_dir, "*.pcd")))[:5]]
            raise FileNotFoundError(
                f"No '*positive*.pcd' test files in {test_dir}. "
                f"First files found: {found} - check the dataset layout "
                f"(normal test samples must contain 'positive' in the filename).")

    elif dataset == "real3d":
        test_dir = os.path.join(root, cls, "test")
        gt_dir = os.path.join(root, cls, "gt")
        test_paths = [s for s in sorted(glob.glob(os.path.join(test_dir, "*.pcd")))
                      if "temp" not in os.path.basename(s)]
        for p in test_paths:
            if "good" in os.path.basename(p):
                pts = read_pcd(p)
                samples.append(TestSample(pts, np.zeros(len(pts), np.float32), 0, p))
            else:
                stem = os.path.splitext(os.path.basename(p))[0]
                txt = os.path.join(gt_dir, stem + ".txt")
                pts, gt = read_txt_cloud(txt)
                samples.append(TestSample(pts, gt, 1, p))
    else:
        raise ValueError(dataset)

    if not samples:
        raise FileNotFoundError(f"No test samples found for {cls} under {root}")
    return samples
