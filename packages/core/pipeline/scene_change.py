"""Scene change detection — the primary frame selection mechanism.

Detects every visual shift: scrolls, page navigations, tab switches, form fills.
A frame is kept when the screen content meaningfully changes from the previous kept frame.
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

RESIZE_DIM = (320, 180)


def _gray_resized(img):
    return cv2.cvtColor(cv2.resize(img, RESIZE_DIM), cv2.COLOR_BGR2GRAY)


def _ssim(g1, g2):
    """Fast mean SSIM between two grayscale images of the same size."""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    g1 = g1.astype(np.float64)
    g2 = g2.astype(np.float64)

    mu1 = cv2.GaussianBlur(g1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(g2, (11, 11), 1.5)
    mu1_sq, mu2_sq, mu12 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    s1 = cv2.GaussianBlur(g1 ** 2, (11, 11), 1.5) - mu1_sq
    s2 = cv2.GaussianBlur(g2 ** 2, (11, 11), 1.5) - mu2_sq
    s12 = cv2.GaussianBlur(g1 * g2, (11, 11), 1.5) - mu12

    num = (2 * mu12 + C1) * (2 * s12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (s1 + s2 + C2)
    return float(np.mean(num / den))


def _hist_corr(img1, img2):
    """Color histogram correlation."""
    h1 = cv2.calcHist([img1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    h2 = cv2.calcHist([img2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(h1, h1)
    cv2.normalize(h2, h2)
    return cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)


def _text_density(gray):
    """Fraction of image covered by text-like edges."""
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 15, 10)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return float(np.sum(closed > 0)) / closed.size


def detect_scene_changes(frames, ssim_change_threshold=0.92):
    """Mark frames where the screen content visually changed.

    Compares each frame to the LAST KEPT frame (not just the previous frame).
    This way scrolling that gradually changes content still triggers a capture
    once enough has changed.

    Args:
        frames: list of frame dicts with 'image_path'
        ssim_change_threshold: SSIM below this vs last kept = new scene (0.92 is sensitive)

    Returns:
        frames with added: scene_change_score, is_scene_change, text_density, visual_importance
    """
    if not frames:
        return frames

    last_kept_gray = None
    last_kept_color = None

    for frame in frames:
        img = cv2.imread(frame["image_path"])
        if img is None:
            frame["scene_change_score"] = 0.0
            frame["is_scene_change"] = False
            frame["text_density"] = 0.0
            frame["visual_importance"] = 0.0
            continue

        small = cv2.resize(img, RESIZE_DIM)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        frame["text_density"] = _text_density(gray)

        if last_kept_gray is None:
            # First frame is always a scene change
            frame["scene_change_score"] = 1.0
            frame["is_scene_change"] = True
            last_kept_gray = gray
            last_kept_color = small
        else:
            ssim = _ssim(last_kept_gray, gray)
            hist = _hist_corr(last_kept_color, small)

            # Combine: lower = more different
            combined_similarity = 0.65 * ssim + 0.35 * hist
            change_score = max(0.0, 1.0 - combined_similarity)

            frame["scene_change_score"] = round(change_score, 4)
            frame["is_scene_change"] = combined_similarity < ssim_change_threshold

            if frame["is_scene_change"]:
                # Update reference to this frame
                last_kept_gray = gray
                last_kept_color = small

        # Composite importance score
        scene_w = frame.get("scene_change_score", 0) * 0.40
        text_w = frame.get("text_density", 0) * 0.20
        rel_w = frame.get("relevance_score", 0) * 0.25
        blur_w = frame.get("blur_score", 0) * 0.15
        frame["visual_importance"] = round(scene_w + text_w + rel_w + blur_w, 4)

    n_changes = sum(1 for f in frames if f.get("is_scene_change"))
    logger.info("Scene changes detected: %d / %d frames", n_changes, len(frames))
    return frames
