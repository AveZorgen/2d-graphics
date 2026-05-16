import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Union
import random
from tqdm import tqdm
from dataclasses import dataclass
import copy

PATCH_SIZE = 128
MIN_CELL_AREA = 10 * 10
CONTEXT_BIN_SIZE = 32


DATA_DIR = Path("data")
BG_DIR = DATA_DIR / "background"
CELL_DIR = DATA_DIR / "cells"


def extract_patches(
    image: np.ndarray, mask: np.ndarray, patch_size: int
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    h, w = image.shape[:2]
    patches = []
    masks = []

    for y in range(0, h - patch_size, patch_size // 2):
        for x in range(0, w - patch_size, patch_size // 2):
            patch_mask = ~mask[y : y + patch_size, x : x + patch_size]
            white_ratio = np.sum(patch_mask < 127) / (patch_size * patch_size)

            if white_ratio < 0.01:
                patch = image[y : y + patch_size, x : x + patch_size]
                patches.append(patch)
                masks.append(patch_mask)

    return patches, masks


@dataclass
class CellRecord:
    image: np.ndarray
    mask: np.ndarray
    context_key: Optional[Tuple[int, int, int]] = None


TrainCellRecord = Tuple[int, int, int, int]


def quantize_color_key(
    color: Union[np.ndarray, Tuple[int, int, int]], bin_size: int = CONTEXT_BIN_SIZE
) -> Tuple[int, int, int]:
    color_arr = np.asarray(color, dtype=np.float32).clip(0, 255)
    return tuple((color_arr // bin_size).astype(int).tolist())


def extract_cells(
    image: np.ndarray, mask: np.ndarray, patch_size: int
) -> Tuple[List[CellRecord], List[TrainCellRecord]]:
    img_h, img_w = image.shape[:2]
    cells: List[CellRecord] = []
    train_cells: List[TrainCellRecord] = []

    bg_pixels = image[mask == 0]
    if bg_pixels.size > 0:
        background_mean = bg_pixels.reshape(-1, 3).mean(axis=0)
    else:
        background_mean = image.reshape(-1, 3).mean(axis=0)
    context_key = quantize_color_key(background_mean)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_CELL_AREA:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        size = min(max(w, h), patch_size)

        cx, cy = x + w // 2, y + h // 2

        train_cells.append((cx / img_w, cy / img_h, w / img_w, h / img_h))

        x1 = cx - size // 2
        y1 = cy - size // 2
        x2 = x1 + size
        y2 = y1 + size
        if x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h:
            continue  # maybe separate these cells on edge as a special case

        cell_img = image[y1:y2, x1:x2]
        cell_mask = mask[y1:y2, x1:x2]

        _, markers, stats, _ = cv2.connectedComponentsWithStats(cell_mask)
        dominated = stats[1:, cv2.CC_STAT_AREA].argmax() + 1
        cell_mask = copy.deepcopy(cell_mask)
        cell_mask[markers != dominated] = 0

        cells.append(
            CellRecord(
                image=cell_img,
                mask=cell_mask,
                context_key=context_key,
            )
        )

    return cells, train_cells


def slice_dataset(
    train_dir="train",
    sample_size: int = 100,
    patch_size: int = PATCH_SIZE,
    verbose=False,
):
    BG_DIR.mkdir(exist_ok=True, parents=True)
    CELL_DIR.mkdir(exist_ok=True, parents=True)
    train_dir = Path(train_dir)

    mask_files = sorted((train_dir / "mask").iterdir())
    orig_files = sorted((train_dir / "original").iterdir())
    indices = random.sample(range(len(orig_files)), min(sample_size, len(orig_files)))

    bg_patches = []
    bg_masks = []
    cell_data: List[CellRecord] = []
    labeled = []

    orig_size = None

    # TODO: make threaded
    for i, idx in tqdm(enumerate(indices), "Slicing dataset", len(indices)):
        image = cv2.imread(
            str(orig_files[idx]), cv2.IMREAD_COLOR_RGB
        )  # str for ultralytics
        mask = cv2.imread(str(mask_files[idx]), cv2.IMREAD_GRAYSCALE).squeeze()

        patches, masks = extract_patches(image, mask, patch_size)
        bg_patches.extend(patches)
        bg_masks.extend(masks)
        cells, train_cells = extract_cells(image, mask, patch_size)
        cell_data.extend(cells)

        if orig_size is None:
            orig_size = mask.shape

        gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        num_cells = len(train_cells)
        labeled.append((gray_img, num_cells, train_cells))

        # TODO:
        # 1. extract patches with multiple cells
        # 1. extract big patches (with multiple cells) as consequence

    print(f"Extracted {len(bg_patches)} background patches and {len(cell_data)} cells")

    if verbose:
        bg_to_write = bg_patches[:5000]
        for i, patch in tqdm(
            enumerate(bg_to_write), "Writing BG patches", len(bg_to_write)
        ):
            cv2.imwrite(
                str(BG_DIR / f"bg_{i:05d}.png"), cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
            )

        cell_to_write = cell_data[:5000]
        for i, record in tqdm(
            enumerate(cell_to_write), "Writing Cell patches", len(cell_to_write)
        ):
            cell, cell_mask = record.image, record.mask
            cv2.imwrite(
                str(CELL_DIR / f"cell_{i:05d}.png"),
                cv2.cvtColor(cell, cv2.COLOR_RGB2BGR),
            )
            cv2.imwrite(str(CELL_DIR / f"cell_{i:05d}_mask.png"), cell_mask)

    if orig_size is None:
        orig_size = (PATCH_SIZE, PATCH_SIZE)

    return bg_patches, cell_data, bg_masks, labeled, orig_size
