import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import random
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid
from joblib import Parallel, delayed
import multiprocessing as mp
import matplotlib.pyplot as plt


TRAIN_DIR = Path("train")
ORIGINAL_DIR = TRAIN_DIR / "original"
MASK_DIR = TRAIN_DIR / "mask"
DATA_DIR = Path("data")
BG_DIR = DATA_DIR / "background"
CELL_DIR = DATA_DIR / "cells"
PATCH_SIZE = 64
TMP_DIR = DATA_DIR / "tmp"


def extract_patches(
    image: np.ndarray, mask: np.ndarray, patch_size: int = PATCH_SIZE
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    h, w = image.shape[:2]
    patches = []
    masks = []

    for y in range(0, h - patch_size, patch_size // 2):
        for x in range(0, w - patch_size, patch_size // 2):
            patch_mask = mask[y : y + patch_size, x : x + patch_size]
            white_ratio = np.sum(patch_mask > 127) / (patch_size * patch_size)

            if white_ratio < 0.1:  # bg patch
                patch = image[y : y + patch_size, x : x + patch_size]
                patches.append(patch)
                masks.append(patch_mask)

    return patches, masks


MIN_CELL_AREA = 100  # r = 5.5
MIN_CELL_AREA = 314  # r ~ 10


def extract_cells(
    image: np.ndarray, mask: np.ndarray, patch_size: int = PATCH_SIZE
) -> List[Tuple[np.ndarray, np.ndarray]]:
    h, w = image.shape[:2]
    cells = []

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_CELL_AREA:
            continue

        x, y, cw, ch = cv2.boundingRect(cnt)
        size = max(cw, ch)
        size = min(size, patch_size)

        cx, cy = x + cw // 2, y + ch // 2

        x1 = max(0, cx - size // 2)
        y1 = max(0, cy - size // 2)
        x2 = min(w, x1 + size)
        y2 = min(h, y1 + size)

        cell_img = image[y1:y2, x1:x2]
        cell_mask = mask[y1:y2, x1:x2]

        if cell_img.size > 0:
            cells.append((cell_img, cell_mask))

    return cells


def slice_dataset(sample_size: int = 100):
    BG_DIR.mkdir(exist_ok=True, parents=True)
    CELL_DIR.mkdir(exist_ok=True, parents=True)

    mask_files = sorted(MASK_DIR.iterdir())
    orig_files = sorted(ORIGINAL_DIR.iterdir())
    indices = random.sample(range(len(orig_files)), min(sample_size, len(orig_files)))

    bg_patches = []
    bg_masks = []
    cell_data = []

    for i, idx in tqdm(enumerate(indices), "Slicing dataset", len(indices)):
        image = cv2.imread(orig_files[idx], cv2.IMREAD_COLOR_RGB)
        mask = cv2.imread(mask_files[idx], cv2.IMREAD_GRAYSCALE)

        patches, masks = extract_patches(image, mask)
        bg_patches.extend(patches)
        bg_masks.extend(masks)
        cell_data.extend(extract_cells(image, mask))

    print(f"Extracted {len(bg_patches)} background patches and {len(cell_data)} cells")

    bg_to_write = bg_patches[:5000]
    for i, patch in tqdm(
        enumerate(bg_to_write), "Writing BG patches", len(bg_to_write)
    ):
        cv2.imwrite(
            str(BG_DIR / f"bg_{i:05d}.png"), cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
        )

    cell_to_write = cell_data[:5000]
    for i, (cell, cell_mask) in tqdm(
        enumerate(cell_to_write), "Writing Cell patches", len(cell_to_write)
    ):
        cv2.imwrite(
            str(CELL_DIR / f"cell_{i:05d}.png"), cv2.cvtColor(cell, cv2.COLOR_RGB2BGR)
        )
        cv2.imwrite(str(CELL_DIR / f"cell_{i:05d}_mask.png"), cell_mask)

    return bg_patches, cell_data, bg_masks


def compute_color_histogram(
    image: np.ndarray, mask: Optional[np.ndarray] = None
) -> np.ndarray:
    hist = cv2.calcHist(
        [image], [0, 1, 2], mask, [32, 32, 32], [0, 256, 0, 256, 0, 256]
    )
    hist = hist / (hist.sum() + 1e-10)
    return hist


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
    bg_patches: List[np.ndarray],
    bg_masks: List[np.ndarray],
    use_synthetic: bool = True,
) -> np.ndarray:
    h, w = target_size
    bg = np.zeros((h, w, 3), dtype=np.uint8)

    if use_synthetic and len(bg_patches) > 0:
        hist = compute_color_histogram(bg_patches[0], cv2.bitwise_not(bg_masks[0]))
        for patch, mask in zip(bg_patches[1:100], bg_masks[1:100]):
            hist += compute_color_histogram(patch, cv2.bitwise_not(mask))
        hist /= 100

        for y in range(0, h, PATCH_SIZE):
            for x in range(0, w, PATCH_SIZE):
                if np.random.rand() < 0.3:
                    b, g, r = sample_from_histogram(hist)
                    bg[y : min(y + PATCH_SIZE, h), x : min(x + PATCH_SIZE, w)] = (
                        b,
                        g,
                        r,
                    )
                elif bg_patches:
                    patch = bg_patches[np.random.randint(len(bg_patches))]
                    patch = cv2.resize(patch, (PATCH_SIZE, PATCH_SIZE))
                    y_end = min(y + PATCH_SIZE, h)
                    x_end = min(x + PATCH_SIZE, w)
                    bg[y:y_end, x:x_end] = patch[: y_end - y, : x_end - x]
    elif bg_patches:
        for y in range(0, h, PATCH_SIZE):
            for x in range(0, w, PATCH_SIZE):
                patch = bg_patches[np.random.randint(len(bg_patches))]
                patch = cv2.resize(patch, (PATCH_SIZE, PATCH_SIZE))
                y_end = min(y + PATCH_SIZE, h)
                x_end = min(x + PATCH_SIZE, w)
                bg[y:y_end, x:x_end] = patch[: y_end - y, : x_end - x]
    else:
        bg.fill(128)

    bg = cv2.GaussianBlur(bg, (5, 5), 0)
    return bg


def generate_synthetic_cell(
    target_size: int, color_hist: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    b, g, r = sample_from_histogram(color_hist)

    radius = np.random.randint(target_size // 4, target_size // 2)
    cell = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    cell[:] = 255
    mask = np.zeros((target_size, target_size), dtype=np.uint8)

    center = (target_size // 2, target_size // 2)
    cv2.circle(cell, center, radius, (b, g, r), -1)
    cv2.circle(mask, center, radius, 255, -1)

    noise = np.random.randint(-20, 20, cell.shape, dtype=np.int16)
    cell = np.clip(cell.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return cell, mask


def seamless_clone(
    src: np.ndarray, dst: np.ndarray, mask: np.ndarray, center: Tuple[int, int]
) -> np.ndarray:
    return cv2.seamlessClone(src, dst, mask, center, cv2.MIXED_CLONE)


NOISE_PARAM_GRID = {
    "gaussian": {
        "mean": [0],
        "stddev": [15, 25, 40],
    },
    "uniform": {
        "low": [-20, -30],
        "high": [20, 30],
    },
}


DENOISE_PARAM_GRID = {
    "median": {
        "ksize": [3, 5, 7],
    },
    "gaussian": {
        "ksize": [(3, 3), (5, 5), (7, 7)],
        "sigmaX": [0, 1],
    },
    "bilateral": {
        "d": [5, 9],
        "sigmaColor": [75],
        "sigmaSpace": [75],
    },
    "nlmeans": {
        "h": [5, 10, 15],
        # "templateWindowSize": [5, 7],
        # "searchWindowSize": [15, 21],
    },
}


def add_noise(image: np.ndarray, noise_type: str = "gaussian", **params) -> np.ndarray:
    if noise_type == "gaussian":
        return cv2.add(image, cv2.randn(np.zeros_like(image), **params))
    elif noise_type == "uniform":
        return cv2.add(image, cv2.randu(np.zeros_like(image), **params))
    else:
        return image.copy()


def denoise_image(image: np.ndarray, method: str = "median", **params) -> np.ndarray:
    if image is None or image.size == 0:
        return image

    if method == "median":
        result = cv2.medianBlur(image, **params)
    elif method == "gaussian":
        result = cv2.GaussianBlur(image, **params)
    elif method == "bilateral":
        result = cv2.bilateralFilter(image, **params)
    elif method == "nlmeans":
        result = cv2.fastNlMeansDenoising(image, **params)
    else:
        result = image

    return result


from skimage.metrics import structural_similarity, mean_squared_error


def calculate_metrics(
    original: np.ndarray, denoised: np.ndarray
) -> Tuple[float, float]:
    mse_gauss = mean_squared_error(original, denoised)
    ssim, _ = structural_similarity(original, denoised, full=True)
    return mse_gauss, 1 - ssim


class BloodCellDataset:
    def __init__(
        self,
        bg_patches: List[np.ndarray],
        bg_masks: List[np.ndarray],
        cell_data: List[Tuple[np.ndarray, np.ndarray]],
        img_size: Tuple[int, int] = (256, 256),
    ):
        self.bg_patches = bg_patches
        self.bg_masks = bg_masks
        self.cell_data = cell_data
        self.img_size = img_size

        self.len = 0
        self.color_hist = self._compute_color_hist()

        self.noise_types_with_params = {
            "gaussian": list(ParameterGrid(NOISE_PARAM_GRID["gaussian"])),
            "uniform": list(ParameterGrid(NOISE_PARAM_GRID["uniform"])),
        }

    def _compute_color_hist(self) -> np.ndarray:
        hist = np.zeros((32, 32, 32), dtype=np.float32)
        count = 0
        for cell, _ in self.cell_data[:100]:
            try:
                h = cv2.calcHist(
                    [cell], [0, 1, 2], None, [32, 32, 32], [0, 256, 0, 256, 0, 256]
                )
                hist += h
                count += 1
            except:
                pass

        if count > 0:
            hist /= count
            hist /= hist.sum() + 1e-10

        self.len = count
        return hist

    def __len__(self):
        return self.len

    def __getitem__(self, idx: int):
        np.random.seed(idx)

        bg = generate_background(
            self.img_size, self.bg_patches, self.bg_masks, use_synthetic=True
        )

        TMP_DIR.mkdir(exist_ok=True, parents=True)
        cv2.imwrite(
            str(TMP_DIR / f"{idx:05d}_0_bg.png"), cv2.cvtColor(bg, cv2.COLOR_RGB2BGR)
        )

        num_cells = np.random.randint(1, 6)

        for i in range(num_cells):
            if self.cell_data and np.random.rand() < 0.7:
                cell, cell_mask = self.cell_data[np.random.randint(len(self.cell_data))]
                cell = cv2.resize(cell, (PATCH_SIZE, PATCH_SIZE))
                cell_mask = cv2.resize(cell_mask, (PATCH_SIZE, PATCH_SIZE))
            else:
                cell, cell_mask = generate_synthetic_cell(PATCH_SIZE, self.color_hist)

            cv2.imwrite(
                str(TMP_DIR / f"{idx:05d}_{i + 1}_cell_{i}.png"),
                cv2.cvtColor(cell, cv2.COLOR_RGB2BGR),
            )

            c1_centr = (PATCH_SIZE // 2, PATCH_SIZE // 2)
            c1_r = PATCH_SIZE // 4
            dx = c1_r + 5
            poly = np.array(
                [
                    [c1_centr[0] - dx, c1_centr[1] - dx],
                    [c1_centr[0] + dx, c1_centr[1] - dx],
                    [c1_centr[0] + dx, c1_centr[1] + dx],
                    [c1_centr[0] - dx, c1_centr[1] + dx],
                ],
                np.int32,
            )
            src_mask = np.zeros_like(cell)
            cv2.fillPoly(src_mask, [poly], (255, 255, 255))

            x = np.random.randint(PATCH_SIZE // 2, self.img_size[1] - PATCH_SIZE // 2)
            y = np.random.randint(PATCH_SIZE // 2, self.img_size[0] - PATCH_SIZE // 2)

            bg = seamless_clone(cell, bg, src_mask, (x, y))

        cv2.imwrite(
            str(TMP_DIR / f"{idx:05d}_{num_cells + 1}_bg_with_cells.png"),
            cv2.cvtColor(bg, cv2.COLOR_RGB2BGR),
        )

        noise_type = random.choice(["gaussian", "uniform"])
        noise_params = random.choice(self.noise_types_with_params[noise_type])

        gray_img = cv2.cvtColor(bg, cv2.COLOR_RGB2GRAY)
        noisy_img = add_noise(bg, noise_type, **noise_params)
        cv2.imwrite(
            str(TMP_DIR / f"{idx:05d}_{num_cells + 2}_noised.png"),
            cv2.cvtColor(noisy_img, cv2.COLOR_RGB2BGR),
        )

        return bg, gray_img, (noisy_img, noise_type, noise_params), num_cells


def process_single_image(args):
    (_, gray_img, _, _), noise_types_with_params, all_denoise_params = args

    local_results = {}

    for noise_type, noise_params_list in noise_types_with_params.items():
        for noise_params in noise_params_list:
            noise_key = f"{noise_type}_{noise_params}"
            local_results[noise_key] = {}

            noisy_img = add_noise(gray_img, noise_type, **noise_params)

            for method, params_list in all_denoise_params.items():
                for params in params_list:
                    param_key = f"{method}_{params}"

                    denoised = denoise_image(noisy_img, method, **params)

                    mse, distortion = calculate_metrics(gray_img, denoised)

                    local_results[noise_key][param_key] = {
                        "mse": mse,
                        "ssim": distortion,
                    }

    return local_results


def run_denoising_experiment(dataset: BloodCellDataset, num_images: int = 100):
    results = {}

    all_denoise_params = {
        method: list(ParameterGrid(DENOISE_PARAM_GRID[method]))
        for method in DENOISE_PARAM_GRID
    }
    noise_types_with_params = dataset.noise_types_with_params

    for noise_type, noise_params_list in noise_types_with_params.items():
        for noise_params in noise_params_list:
            key = f"{noise_type}_{noise_params}"
            results[key] = {}

            for method, params_list in all_denoise_params.items():
                results[key][method] = {
                    "mse": [],
                    "ssim": [],
                }

                for params in params_list:
                    results[key][f"{method}_{params}"] = {
                        "mse": [],
                        "ssim": [],
                    }

    args_list = [
        (dataset[i], noise_types_with_params, all_denoise_params)
        for i in range(num_images)
    ]

    n_jobs = min(mp.cpu_count(), 8)
    print(f"Running denoising experiment with {n_jobs} parallel jobs...")

    parallel_results = Parallel(n_jobs=n_jobs)(
        delayed(process_single_image)(args)
        for args in tqdm(args_list, desc="Processing images")
    )
    # parallel_results = (
    #     process_single_image(args) for args in tqdm(args_list, desc="Processing images")
    # )

    for result in parallel_results:
        for noise_key, method_dict in result.items():
            for method_key, metrics in method_dict.items():
                results[noise_key][method_key]["mse"].append(metrics["mse"])
                results[noise_key][method_key]["ssim"].append(metrics["ssim"])

    print("\n" + "=" * 60)
    print("DENOISING RESULTS")
    print("=" * 60)

    best_results = {}

    for noise_key in results.keys():
        print(f"\n--- {noise_key} ---")
        best_mse = float("inf")
        best_config = None

        for method_key, metrics in results[noise_key].items():
            if metrics["mse"]:
                avg_mse = np.mean(metrics["mse"])
                avg_ssim = 1 - np.mean(metrics["ssim"])
                print(f"  {method_key}: MSE={avg_mse:.2f}, 1-SSIM={avg_ssim:.4f}")

                if avg_mse < best_mse:
                    best_mse = avg_mse
                    best_config = method_key

        best_results[noise_key] = {"method": best_config, "mse": best_mse}
        print(f"  BEST: {best_config} with MSE={best_mse:.2f}")

    print("\n" + "=" * 60)
    print("SUMMARY - Best methods per noise type:")
    print("=" * 60)
    for noise_key, info in best_results.items():
        print(f"  {noise_key}: {info['method']} (MSE={info['mse']:.2f})")

    return results


if __name__ == "__main__":
    bg_patches, cell_data, bg_masks = slice_dataset(sample_size=100)

    print("Creating dataset...")
    dataset = BloodCellDataset(bg_patches, bg_masks, cell_data, img_size=(256, 256))

    results = run_denoising_experiment(dataset, num_images=100)
