import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import random
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import NearestNeighbors

from .preprocess import CellRecord, quantize_color_key
from .noise import NOISE_PARAM_GRID, add_noise

# NOTE: implementation may differ from hw2.ipynb


DATA_DIR = Path("data")
TMP_DIR = DATA_DIR / "tmp"


class BloodCellDataset:
    def __init__(
        self,
        orig_size: Tuple[int, int],
        img_size: Tuple[int, int] = (256, 256),
        n=100,
        verbose=False,
    ):
        self.img_size = img_size
        self.orig_size = orig_size

        self.noise_types_with_params = {
            alg: list(ParameterGrid(NOISE_PARAM_GRID[alg])) for alg in NOISE_PARAM_GRID
        }

        self.verbose = verbose
        if verbose:
            TMP_DIR.mkdir(exist_ok=True, parents=True)

        self.bg_patches = []
        self.bg_masks = []
        self.cell_data = []
        self.color_hist = None
        self.background_hist = None
        self.cells_by_context = {}
        self.color_hist_by_context = {}
        self.patch_count = 0
        self.nbrs = None
        self.cell_num_p = [1.0]
        self.cell_num_v = [1.0, 1.0]

        self.patch_size = self.img_size

        self.data = {}
        self.n = n

    def fit(
        self,
        data: Tuple[
            List[np.ndarray],
            List[np.ndarray],
            List[CellRecord],
            List[Tuple[np.array, int]],
        ],
        max_samples: int = 5000,
    ):
        bg_patches, bg_masks, cell_data, labeled_orig = data
        if len(bg_patches) > max_samples:
            indices = random.sample(range(len(bg_patches)), max_samples)
            bg_patches = [bg_patches[i] for i in indices]
            bg_masks = [bg_masks[i] for i in indices]

        orig_patch_size = bg_patches[0].shape[0]  # square

        self.bg_patches = bg_patches
        self.bg_masks = bg_masks
        self.cell_data = [(record.image, record.mask) for record in cell_data]

        self.color_hist = BloodCellDataset._compute_color_hist(self.cell_data)
        self.background_hist = BloodCellDataset._compute_color_hist(
            list(zip(self.bg_patches, self.bg_masks))
        )
        self.cells_by_context = BloodCellDataset._build_context_index(cell_data)
        self.color_hist_by_context = BloodCellDataset._build_context_histograms(
            self.cells_by_context
        )

        mean_colors = np.mean(self.bg_patches, axis=(1, 2))

        patches_h = (self.orig_size[0] + orig_patch_size - 1) // orig_patch_size
        patches_w = (self.orig_size[1] + orig_patch_size - 1) // orig_patch_size

        self.patch_count = max(1, patches_h * patches_w)
        # * 2 for variety
        self.nbrs = NearestNeighbors(
            n_neighbors=self.patch_count * 2, algorithm="ball_tree"
        ).fit(
            mean_colors
        )  # will search for n_ closest to given color

        patch_size_h = (self.img_size[0] + patches_h - 1) // patches_h
        patch_size_w = (self.img_size[1] + patches_w - 1) // patches_w
        self.patch_size = patch_size_h, patch_size_w

        h, v = np.histogram([n for _, n in labeled_orig], bins=50)
        h = h / h.sum()
        self.cell_num_p = h
        self.cell_num_v = v

        self.data = {}

    @staticmethod
    def _compute_color_hist(data: list, max_samples: int = 100) -> np.ndarray:
        hist = np.zeros((32, 32, 32), dtype=np.float32)
        if not data:
            hist += 1
            hist /= hist.sum()
            return hist

        sample_count = min(len(data), max(1, max_samples))
        if len(data) > sample_count:
            sampled_data = random.sample(data, sample_count)
        else:
            sampled_data = data

        for image, mask in sampled_data:
            h = cv2.calcHist(
                [image], [0, 1, 2], mask, [32, 32, 32], [0, 256, 0, 256, 0, 256]
            )
            h_sum = h.sum()
            if h_sum > 0:
                hist += h / h_sum
        if hist.sum() == 0:
            hist += 1

        hist /= hist.sum()

        return hist

    def __len__(self):
        # raise NotImplementedError()
        return self.n

    @staticmethod
    def _build_context_index(
        cell_data: List[CellRecord],
    ) -> Dict[Tuple[int, int, int], List[Tuple[np.ndarray, np.ndarray]]]:
        context_index = {}
        for record in cell_data:
            context_index.setdefault(record.context_key, []).append(
                (record.image, record.mask)
            )

        return context_index

    @staticmethod
    def _build_context_histograms(
        cells_by_context: Dict[
            Tuple[int, int, int], List[Tuple[np.ndarray, np.ndarray]]
        ],
    ) -> Dict[Tuple[int, int, int], np.ndarray]:
        return {
            context_key: BloodCellDataset._compute_color_hist(records)
            for context_key, records in cells_by_context.items()
        }

    def _get_real_cell_source(
        self, context_key: Tuple[int, int, int]
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        return self.cells_by_context.get(context_key, self.cell_data)

    def _get_synthetic_color_hist(
        self, context_key: Tuple[int, int, int]
    ) -> np.ndarray:
        return self.color_hist_by_context.get(context_key, self.color_hist)

    def _estimate_num_cells(self) -> int:
        idx = np.random.choice(len(self.cell_num_p), p=self.cell_num_p)
        return np.random.randint(self.cell_num_v[idx], self.cell_num_v[idx + 1])

    def __getitem__(self, idx: int):
        # if idx in self.data:
        # return self.data[idx]
        np.random.seed(idx)
        random.seed(idx)

        bg, base_color = generate_background(
            self.img_size,
            self.patch_size,
            self.bg_patches,
            self.background_hist,
            self.nbrs,
        )
        context_key = quantize_color_key(base_color)

        if self.verbose:
            cv2.imwrite(str(TMP_DIR / f"{idx:05d}_0_bg.png"), bg)

        num_cells = self._estimate_num_cells()

        composite_src = bg.copy()
        composite_mask = np.zeros(bg.shape[:2], dtype=np.uint8)

        for i in range(num_cells):
            # NOTE: key idea: not overfit future training but create representative dataset
            if self.cell_data and np.random.rand() < 0.7:
                source = self._get_real_cell_source(context_key)
                cell, cell_mask = source[np.random.randint(len(source))]
                cell = cv2.resize(cell, self.patch_size[::-1])
                cell_mask = cv2.resize(cell_mask, self.patch_size[::-1])
            else:
                synthetic_hist = self._get_synthetic_color_hist(context_key)
                cell, cell_mask = generate_synthetic_cell(
                    self.patch_size, synthetic_hist
                )

            if self.verbose:
                cv2.imwrite(str(TMP_DIR / f"{idx:05d}_{i + 1}_cell_{i}.png"), cell)

            h, w = self.patch_size
            x = np.random.randint(w // 2, self.img_size[1] - w // 2)
            y = np.random.randint(h // 2, self.img_size[0] - h // 2)

            x1 = x - w // 2
            x2 = x + w // 2
            y1 = y - h // 2
            y2 = y + h // 2

            bg_roi = bg[y1:y2, x1:x2]

            src_roi = cell.copy()
            src_roi[cell_mask == 0] = bg_roi[cell_mask == 0]

            alpha = 0.7

            bg[y1:y2, x1:x2] = cv2.addWeighted(src_roi, alpha, bg_roi, 1 - alpha, 0)
            composite_mask[y1:y2, x1:x2] = np.maximum(
                composite_mask[y1:y2, x1:x2], cell_mask
            )

        if np.any(composite_mask):
            center = (self.img_size[1] // 2, self.img_size[0] // 2)
            bg = cv2.seamlessClone(
                composite_src, bg, composite_mask.copy(), center, cv2.MIXED_CLONE
            )

        if self.verbose:
            cv2.imwrite(
                str(TMP_DIR / f"{idx:05d}_{num_cells + 1}_bg_with_cells.png"), bg
            )

        noise_type = random.choice(["gaussian", "uniform"])
        noise_params = random.choice(self.noise_types_with_params[noise_type])

        gray_img = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
        noisy_img = add_noise(gray_img, noise_type, **noise_params)
        if self.verbose:
            cv2.imwrite(
                str(TMP_DIR / f"{idx:05d}_{num_cells + 2}_noised.png"), noisy_img
            )

        return bg, gray_img, (noisy_img, noise_type, noise_params), num_cells
        # self.data[idx] = bg, gray_img, (noisy_img, noise_type, noise_params), num_cells
        # return self.data[idx]


def sample_from_histogram(hist: np.ndarray) -> Tuple[int, int, int]:
    flat = hist.flatten()
    idx = np.random.choice(len(flat), p=flat)
    coords = np.unravel_index(idx, hist.shape)
    b = coords[0] * 8 + np.random.randint(0, 8)
    g = coords[1] * 8 + np.random.randint(0, 8)
    r = coords[2] * 8 + np.random.randint(0, 8)
    return int(b), int(g), int(r)


def generate_background(
    target_size: Tuple[int, int],
    patch_size: Tuple[int, int],
    bg_patches: List[np.ndarray],
    bg_hist: np.ndarray,
    sampler: NearestNeighbors,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = target_size
    p_h, p_w = patch_size
    bg = np.zeros((h, w, 3), dtype=np.uint8)

    base_color = sample_from_histogram(bg_hist)
    indices = sampler.kneighbors([base_color], return_distance=False)[0]
    random.shuffle(indices)
    patches = [bg_patches[i] for i in indices]
    c = 0

    for y in range(0, h, p_h):
        for x in range(0, w, p_w):
            y_end = min(y + p_h, h)
            x_end = min(x + p_w, w)

            if np.random.rand() < 0.3:
                color = base_color
            else:
                patch = patches[c]
                patch = cv2.resize(patch, patch_size[::-1])
                color = patch[: y_end - y, : x_end - x]
                c += 1

            bg[y:y_end, x:x_end] = color

    bg = cv2.GaussianBlur(bg, (5, 5), 0)
    return bg, base_color


def generate_synthetic_cell(
    target_size: Tuple[int, int], color_hist: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    color = sample_from_histogram(color_hist)

    h, w = target_size
    low = min(h, w)
    radius = np.random.randint(low // 4, low // 2)
    cell = np.zeros((h, w, 3), dtype=np.uint8)
    cell[:] = 255
    mask = np.zeros((h, w), dtype=np.uint8)

    center = (h // 2, w // 2)
    cv2.circle(cell, center, radius, color, -1)
    cv2.circle(mask, center, radius, 255, -1)

    noise = np.random.randint(-20, 20, cell.shape, dtype=np.int16)
    cell = np.clip(cell.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return cell, mask
