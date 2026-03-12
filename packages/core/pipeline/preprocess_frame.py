"""
Frame preprocessing module for PolicyCapture Local.

Computes quality metrics (blur, stability) for extracted frames.
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Empirical maximum Laplacian variance used for normalization.
# Typical sharp screenshots produce values around 1000-3000; cap at 5000.
_LAPLACIAN_VARIANCE_CAP = 5000.0


def compute_blur_score(image: np.ndarray) -> float:
    """
    Compute a sharpness score for an image using Laplacian variance.

    Args:
        image: BGR or grayscale image as a NumPy array.

    Returns:
        Float in [0, 1] where higher values indicate a sharper image.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    score = min(float(laplacian_var) / _LAPLACIAN_VARIANCE_CAP, 1.0)
    return round(score, 4)


def compute_stability_score(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute a structural similarity score between two images.

    Uses a simplified mean-squared-error approach normalized to a 0-1
    similarity value.  For production use, consider skimage.metrics.structural_similarity.

    Args:
        img1: First image (BGR or grayscale).
        img2: Second image (BGR or grayscale).

    Returns:
        Float in [0, 1] where 1 means the images are identical.
    """
    # Convert to grayscale if needed
    if len(img1.shape) == 3:
        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    else:
        g1 = img1

    if len(img2.shape) == 3:
        g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        g2 = img2

    # Resize to common dimensions if they differ
    if g1.shape != g2.shape:
        h = min(g1.shape[0], g2.shape[0])
        w = min(g1.shape[1], g2.shape[1])
        g1 = cv2.resize(g1, (w, h))
        g2 = cv2.resize(g2, (w, h))

    mse = np.mean((g1.astype(np.float64) - g2.astype(np.float64)) ** 2)

    # Convert MSE to a 0-1 similarity.  Max possible MSE for uint8 is 255^2.
    max_mse = 255.0 ** 2
    similarity = 1.0 - (mse / max_mse)
    return round(max(0.0, min(1.0, similarity)), 4)


def preprocess_frame(image_path: str) -> dict:
    """
    Preprocess a single frame image and compute quality metrics.

    Args:
        image_path: Path to the frame image file.

    Returns:
        dict with keys:
            blur_score (float): Sharpness score in [0, 1].
            is_sharp (bool): True if blur_score >= 0.15.
            dimensions (tuple[int, int]): (width, height) of the image.
    """
    image = cv2.imread(image_path)
    if image is None:
        logger.error("Could not read image: %s", image_path)
        return {
            "blur_score": 0.0,
            "is_sharp": False,
            "dimensions": (0, 0),
        }

    blur_score = compute_blur_score(image)
    h, w = image.shape[:2]

    result = {
        "blur_score": blur_score,
        "is_sharp": blur_score >= 0.15,
        "dimensions": (w, h),
    }

    logger.debug(
        "Preprocessed %s — blur_score=%.4f, is_sharp=%s, dimensions=%s",
        image_path,
        result["blur_score"],
        result["is_sharp"],
        result["dimensions"],
    )
    return result
