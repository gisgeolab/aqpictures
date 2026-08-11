import csv
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from matplotlib.colors import rgb_to_hsv
from PIL import Image


IMG_EXTS = {
    ".jpg",
    ".jpeg",
    ".JPG",
    ".JPEG",
    ".png",
    ".PNG",
}

NAME_RE = re.compile(
    r"^(?P<date>\d{8})-(?P<hour>\d{4})$"
)


# ============================================================
# Basic image loading and ROI processing
# ============================================================

def load_rgb_array(image_path):
    """
    Load an image and return an RGB NumPy array.
    """
    image_path = Path(image_path)

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        return np.asarray(img, dtype=np.uint8)


def crop_roi(arr, top, left, height, width):
    """
    Crop a fixed rectangular ROI from an RGB image array.
    """
    return arr[
        top:top + height,
        left:left + width,
        :
    ]


def rgb_to_grayscale(arr):
    """
    Convert RGB image to grayscale using luminance weights.

    Output range: approximately 0-255.
    """
    if arr.size == 0:
        return np.array([], dtype=np.float64)

    arr_float = arr.astype(np.float64)

    gray = (
        0.299 * arr_float[:, :, 0]
        + 0.587 * arr_float[:, :, 1]
        + 0.114 * arr_float[:, :, 2]
    )

    return gray


def get_sky_subregion(arr, sky_fraction=0.50):
    """
    Extract the upper part of the ROI as a sky-dominant subregion.

    Parameters
    ----------
    arr : np.ndarray
        ROI RGB array.
    sky_fraction : float
        Fraction of the ROI height used as sky region.
        For example, 0.50 means the upper 50% of the ROI.

    Returns
    -------
    np.ndarray
        Sky-region RGB array.
    """
    if arr.size == 0:
        return arr

    if not 0 < sky_fraction <= 1:
        raise ValueError(
            f"sky_fraction must be between 0 and 1, got {sky_fraction}"
        )

    sky_height = max(
        1,
        int(round(arr.shape[0] * sky_fraction))
    )

    return arr[:sky_height, :, :]


# ============================================================
# Color and brightness features
# ============================================================

def mean_rgb(arr):
    """
    Calculate mean values of the red, green, and blue channels.
    """
    if arr.size == 0:
        return np.nan, np.nan, np.nan

    arr_float = arr.astype(np.float64)

    r = arr_float[:, :, 0].mean()
    g = arr_float[:, :, 1].mean()
    b = arr_float[:, :, 2].mean()

    return float(r), float(g), float(b)


def std_rgb(arr):
    """
    Calculate standard deviations of the red, green, and blue
    channels.
    """
    if arr.size == 0:
        return np.nan, np.nan, np.nan

    arr_float = arr.astype(np.float64)

    r_std = arr_float[:, :, 0].std()
    g_std = arr_float[:, :, 1].std()
    b_std = arr_float[:, :, 2].std()

    return (
        float(r_std),
        float(g_std),
        float(b_std),
    )


def mean_saturation(arr):
    """
    Calculate the mean HSV saturation value.

    Output range: 0-1.
    """
    if arr.size == 0:
        return np.nan

    arr_norm = arr.astype(np.float64) / 255.0
    hsv = rgb_to_hsv(arr_norm)

    saturation = hsv[:, :, 1]

    return float(saturation.mean())


def mean_value(arr):
    """
    Calculate the mean HSV Value channel.

    Output range: 0-1.
    """
    if arr.size == 0:
        return np.nan

    arr_norm = arr.astype(np.float64) / 255.0
    hsv = rgb_to_hsv(arr_norm)

    value = hsv[:, :, 2]

    return float(value.mean())


def blue_red_ratio(arr):
    """
    Calculate the ratio between mean blue and mean red values.
    """
    if arr.size == 0:
        return np.nan

    arr_float = arr.astype(np.float64)

    r_mean = arr_float[:, :, 0].mean()
    b_mean = arr_float[:, :, 2].mean()

    eps = 1e-6

    return float(
        b_mean / (r_mean + eps)
    )


def image_colorfulness(arr):
    """
    Calculate image colorfulness using the Hasler-Suesstrunk
    formulation.

    Larger values indicate greater color variation and stronger
    colorfulness. Lower values indicate a more gray or desaturated
    scene.
    """
    if arr.size == 0:
        return np.nan

    arr_float = arr.astype(np.float64)

    r = arr_float[:, :, 0]
    g = arr_float[:, :, 1]
    b = arr_float[:, :, 2]

    rg = r - g
    yb = 0.5 * (r + g) - b

    std_rg = np.std(rg)
    std_yb = np.std(yb)

    mean_rg = np.mean(rg)
    mean_yb = np.mean(yb)

    std_component = np.sqrt(
        std_rg ** 2 + std_yb ** 2
    )

    mean_component = np.sqrt(
        mean_rg ** 2 + mean_yb ** 2
    )

    colorfulness = (
        std_component
        + 0.3 * mean_component
    )

    return float(colorfulness)


# ============================================================
# Contrast and texture features
# ============================================================

def image_contrast(arr):
    """
    Calculate global grayscale contrast as the standard deviation
    of grayscale pixel values.
    """
    if arr.size == 0:
        return np.nan

    gray = rgb_to_grayscale(arr)

    return float(gray.std())


def grayscale_entropy(arr, bins=256):
    """
    Calculate Shannon entropy of the grayscale histogram.

    Higher values generally indicate more complex grayscale
    information.
    """
    if arr.size == 0:
        return np.nan

    gray = rgb_to_grayscale(arr)

    histogram, _ = np.histogram(
        gray,
        bins=bins,
        range=(0, 256),
    )

    histogram = histogram.astype(np.float64)
    total = histogram.sum()

    if total == 0:
        return np.nan

    probabilities = histogram / total
    probabilities = probabilities[
        probabilities > 0
    ]

    entropy = -np.sum(
        probabilities * np.log2(probabilities)
    )

    return float(entropy)


def laplacian_variance(arr):
    """
    Calculate the variance of the image Laplacian.

    Larger values generally indicate sharper edges and more
    high-frequency detail.
    """
    if arr.size == 0:
        return np.nan

    gray = rgb_to_grayscale(arr)

    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return np.nan

    padded = np.pad(
        gray,
        pad_width=1,
        mode="edge",
    )

    laplacian = (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
        - 4.0 * padded[1:-1, 1:-1]
    )

    return float(
        np.var(laplacian)
    )


def mean_gradient_magnitude(arr):
    """
    Calculate the mean grayscale gradient magnitude.

    Larger values indicate stronger average image transitions and
    clearer structural boundaries.
    """
    if arr.size == 0:
        return np.nan

    gray = rgb_to_grayscale(arr)

    if gray.shape[0] < 2 or gray.shape[1] < 2:
        return np.nan

    gradient_y, gradient_x = np.gradient(gray)

    gradient_magnitude = np.sqrt(
        gradient_x ** 2
        + gradient_y ** 2
    )

    return float(
        gradient_magnitude.mean()
    )


def local_contrast(arr, block_size=32):
    """
    Calculate local contrast by dividing the grayscale ROI into
    blocks, calculating the standard deviation of each block, and
    averaging the block-level standard deviations.

    Parameters
    ----------
    arr : np.ndarray
        ROI RGB array.
    block_size : int
        Height and width of each local block.
    """
    if arr.size == 0:
        return np.nan

    if block_size <= 0:
        raise ValueError(
            f"block_size must be positive, got {block_size}"
        )

    gray = rgb_to_grayscale(arr)

    height, width = gray.shape
    block_contrasts = []

    for row_start in range(0, height, block_size):
        row_end = min(
            row_start + block_size,
            height,
        )

        for col_start in range(0, width, block_size):
            col_end = min(
                col_start + block_size,
                width,
            )

            block = gray[
                row_start:row_end,
                col_start:col_end,
            ]

            if block.size > 0:
                block_contrasts.append(
                    float(block.std())
                )

    if not block_contrasts:
        return np.nan

    return float(
        np.mean(block_contrasts)
    )


# ============================================================
# Haze and visibility features
# ============================================================

def dark_pixel_ratio(arr, threshold=50):
    """
    Calculate the proportion of dark grayscale pixels.

    Parameters
    ----------
    arr : np.ndarray
        ROI RGB array.
    threshold : float
        Grayscale threshold in the 0-255 range.

    Returns
    -------
    float
        Proportion of pixels whose grayscale intensity is lower
        than the threshold. Output range: 0-1.
    """
    if arr.size == 0:
        return np.nan

    if not 0 <= threshold <= 255:
        raise ValueError(
            f"threshold must be between 0 and 255, got {threshold}"
        )

    gray = rgb_to_grayscale(arr)

    ratio = np.mean(
        gray < threshold
    )

    return float(ratio)


def sky_brightness(arr, sky_fraction=0.50):
    """
    Calculate mean HSV Value in the upper sky-dominant part of
    the ROI.

    Output range: 0-1.
    """
    if arr.size == 0:
        return np.nan

    sky_arr = get_sky_subregion(
        arr,
        sky_fraction=sky_fraction,
    )

    return mean_value(sky_arr)


def sky_luminance_gradient(arr, sky_fraction=0.50):
    """
    Calculate the vertical luminance gradient in the upper
    sky-dominant part of the ROI.

    The value is the slope from a linear regression between
    normalized vertical position and mean row luminance.

    Positive value:
        Lower rows are brighter than upper rows.

    Negative value:
        Lower rows are darker than upper rows.
    """
    if arr.size == 0:
        return np.nan

    sky_arr = get_sky_subregion(
        arr,
        sky_fraction=sky_fraction,
    )

    gray = rgb_to_grayscale(sky_arr)

    if gray.shape[0] < 2:
        return np.nan

    row_luminance = gray.mean(axis=1)

    vertical_position = np.linspace(
        0.0,
        1.0,
        num=len(row_luminance),
    )

    slope = np.polyfit(
        vertical_position,
        row_luminance,
        deg=1,
    )[0]

    return float(slope)


# ============================================================
# Complete ROI feature extraction
# ============================================================

def extract_roi_features(
    image_path,
    roi,
    sky_fraction=0.50,
    dark_threshold=50,
    local_block_size=32,
):
    """
    Extract all image features from the selected ROI.
    """
    image_path = Path(image_path)
    arr = load_rgb_array(image_path)

    top = int(roi["top"])
    left = int(roi["left"])
    height = int(roi["height"])
    width = int(roi["width"])

    if top < 0 or left < 0:
        raise ValueError(
            f"ROI top and left must be non-negative: {roi}"
        )

    if height <= 0 or width <= 0:
        raise ValueError(
            f"ROI height and width must be positive: {roi}"
        )

    if arr.ndim != 3:
        raise ValueError(
            f"Expected a three-dimensional RGB array, "
            f"got shape={arr.shape}"
        )

    image_height, image_width, channels = arr.shape

    if (
        channels != 3
        or image_height < top + height
        or image_width < left + width
    ):
        raise ValueError(
            f"Image too small for ROI cropping: "
            f"{image_path.name}, "
            f"image shape={arr.shape}, "
            f"roi={roi}"
        )

    roi_arr = crop_roi(
        arr,
        top,
        left,
        height,
        width,
    )

    if roi_arr.size == 0:
        raise ValueError(
            f"ROI is empty for image: {image_path.name}"
        )

    # Color and brightness
    r_roi, g_roi, b_roi = mean_rgb(roi_arr)

    r_std, g_std, b_std = std_rgb(roi_arr)

    s_mean = mean_saturation(roi_arr)
    v_mean = mean_value(roi_arr)

    br_ratio = blue_red_ratio(roi_arr)

    colorfulness = image_colorfulness(
        roi_arr
    )

    # Contrast and texture
    contrast = image_contrast(roi_arr)

    gray_entropy = grayscale_entropy(
        roi_arr
    )

    laplacian_var = laplacian_variance(
        roi_arr
    )

    gradient_magnitude = mean_gradient_magnitude(
        roi_arr
    )

    local_contrast_value = local_contrast(
        roi_arr,
        block_size=local_block_size,
    )

    # Haze and visibility
    dark_ratio = dark_pixel_ratio(
        roi_arr,
        threshold=dark_threshold,
    )

    sky_brightness_value = sky_brightness(
        roi_arr,
        sky_fraction=sky_fraction,
    )

    sky_gradient = sky_luminance_gradient(
        roi_arr,
        sky_fraction=sky_fraction,
    )

    return {
        # Color and brightness features
        "R_roi": r_roi,
        "G_roi": g_roi,
        "B_roi": b_roi,
        "R_std": r_std,
        "G_std": g_std,
        "B_std": b_std,
        "S_mean": s_mean,
        "V_mean": v_mean,
        "colorfulness": colorfulness,
        "sky_brightness": sky_brightness_value,

        # Contrast and texture features
        "contrast": contrast,
        "gray_entropy": gray_entropy,
        "laplacian_variance": laplacian_var,
        "mean_gradient_magnitude": gradient_magnitude,
        "local_contrast": local_contrast_value,

        # Haze and visibility features
        "B_R_ratio": br_ratio,
        "dark_pixel_ratio": dark_ratio,
        "sky_luminance_gradient": sky_gradient,
    }


# ============================================================
# Batch feature extraction and CSV output
# ============================================================

def extract_image_features(
    image_dir,
    output_csv,
    roi,
    project_root=None,
    image_timezone="Europe/Rome",
    output_timezone="UTC",
    sky_fraction=0.50,
    dark_threshold=50,
    local_block_size=32,
):
    """
    Extract ROI image features from all valid image files and save
    the resulting table as a CSV file.

    Expected filename format:
        YYYYMMDD-HHMM.jpg

    Example:
        20250615-1200.jpg
    """
    image_dir = Path(image_dir)
    output_csv = Path(output_csv)

    if project_root is not None:
        project_root = Path(project_root)

    if not image_dir.exists():
        raise FileNotFoundError(
            f"Image directory does not exist: {image_dir}"
        )

    if not image_dir.is_dir():
        raise NotADirectoryError(
            f"image_dir is not a directory: {image_dir}"
        )

    rows = []
    skipped_files = []

    for image_path in sorted(image_dir.rglob("*")):
        if not (
            image_path.is_file()
            and image_path.suffix in IMG_EXTS
        ):
            continue

        stem = image_path.stem

        if not NAME_RE.match(stem):
            skipped_files.append(
                (
                    str(image_path),
                    "Invalid filename format",
                )
            )
            continue

        try:
            dt_local = datetime.strptime(
                stem,
                "%Y%m%d-%H%M",
            ).replace(
                tzinfo=ZoneInfo(image_timezone)
            )

            dt_output = dt_local.astimezone(
                ZoneInfo(output_timezone)
            )

        except (
            ValueError,
            KeyError,
        ) as error:
            skipped_files.append(
                (
                    str(image_path),
                    f"Datetime parsing failed: {error}",
                )
            )
            continue

        datetime_csv = dt_output.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        try:
            features = extract_roi_features(
                image_path=image_path,
                roi=roi,
                sky_fraction=sky_fraction,
                dark_threshold=dark_threshold,
                local_block_size=local_block_size,
            )

        except Exception as error:
            skipped_files.append(
                (
                    str(image_path),
                    str(error),
                )
            )
            continue

        if project_root is not None:
            try:
                image_path_value = str(
                    image_path.relative_to(
                        project_root
                    )
                )
            except ValueError:
                image_path_value = str(
                    image_path
                )
        else:
            image_path_value = str(
                image_path
            )

        rows.append(
            {
                "datetime": datetime_csv,
                **features,
                "image_path": image_path_value,
            }
        )

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "datetime",

        # Color and brightness features
        "R_roi",
        "G_roi",
        "B_roi",
        "R_std",
        "G_std",
        "B_std",
        "S_mean",
        "V_mean",
        "colorfulness",
        "sky_brightness",

        # Contrast and texture features
        "contrast",
        "gray_entropy",
        "laplacian_variance",
        "mean_gradient_magnitude",
        "local_contrast",

        # Haze and visibility features
        "B_R_ratio",
        "dark_pixel_ratio",
        "sky_luminance_gradient",

        "image_path",
    ]

    with output_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    return rows, skipped_files