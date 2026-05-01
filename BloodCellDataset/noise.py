import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import random
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid
from joblib import Parallel, delayed
import multiprocessing as mp
import matplotlib.pyplot as plt
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import copy
from sklearn.neighbors import NearestNeighbors
from skimage.metrics import structural_similarity

# NOTE: this is only partial port of noise experiments form hw2.ipynb

NOISE_PARAM_GRID_FEW = {
    "gaussian": {
        "mean": [0],
        "stddev": [5, 25],
    },
    "uniform": {
        "low": [-10, -60],
        "high": [40],
    },
    "periodic": {
        "A": [20, 100],
        "B": [20, 100],
        "angle": [0, 45, 90],
    },
}
NOISE_PARAM_GRID = NOISE_PARAM_GRID_FEW


def add_noise(image: np.ndarray, noise_type: str = "gaussian", **params) -> np.ndarray:
    if noise_type == "gaussian":
        return cv2.add(image, cv2.randn(np.zeros_like(image), **params))
    elif noise_type == "uniform":
        return cv2.add(image, cv2.randu(np.zeros_like(image), **params))
    elif noise_type == "periodic":
        A = params.get("A", 30)
        B = params.get("B", 20)
        angle = params.get("angle", 0)

        h, w = image.shape[:2]
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        theta = np.radians(angle)
        x_rot = x_coords * np.cos(theta) + y_coords * np.sin(theta)

        noise = np.sin(x_rot / B * 2 * np.pi) * A

        return cv2.add(image.astype(np.float32), noise.astype(np.float32)).astype(
            np.uint8
        )
    else:
        return image.copy()
