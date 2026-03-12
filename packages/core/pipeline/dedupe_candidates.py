"""
Candidate deduplication module for PolicyCapture Local.

Removes near-duplicate frames based on perceptual image hashing.
"""

# TODO: Can enhance with perceptual hashing (pHash) or learned embeddings

import logging
from typing import List

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def compute_image_hash(image_path: str) -> str:
    """
    Compute an average hash for an image.

    The image is resized to 8x8 grayscale, and each pixel is compared to the
    mean to produce a 64-bit binary hash string.

    Args:
        image_path: Path to the image file.

    Returns:
        64-character string of '0' and '1' representing the hash,
        or empty string on failure.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        logger.error("Could not read image for hashing: %s", image_path)
        return ""

    resized = cv2.resize(image, (8, 8), interpolation=cv2.INTER_AREA)
    mean_val = resized.mean()
    bits = (resized > mean_val).astype(np.uint8).flatten()
    return "".join(str(b) for b in bits)


def compute_hash_similarity(hash1: str, hash2: str) -> float:
    """
    Compute similarity between two image hashes using Hamming distance.

    Args:
        hash1: 64-character binary hash string.
        hash2: 64-character binary hash string.

    Returns:
        Float in [0, 1] where 1.0 means identical hashes.
    """
    if not hash1 or not hash2:
        return 0.0
    if len(hash1) != len(hash2):
        return 0.0

    hamming = sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
    similarity = 1.0 - (hamming / len(hash1))
    return round(similarity, 4)


def dedupe_candidates(
    candidates: List[dict],
    similarity_threshold: float = 0.92,
) -> List[dict]:
    """
    Remove near-duplicate frames, keeping the higher-scored candidate.

    Each candidate dict must contain at least:
        image_path (str): Path to the frame image.
        composite_score (float): Quality/relevance score.

    Args:
        candidates: List of candidate frame dicts.
        similarity_threshold: Hash similarity above which two frames are
                              considered duplicates (0-1).

    Returns:
        Deduplicated list of candidate dicts.
    """
    if not candidates:
        return []

    # Sort by composite_score descending so we keep the best first
    sorted_cands = sorted(
        candidates,
        key=lambda c: c.get("composite_score", 0.0),
        reverse=True,
    )

    # Compute hashes
    for cand in sorted_cands:
        if "image_hash" not in cand:
            cand["image_hash"] = compute_image_hash(cand.get("image_path", ""))

    kept: List[dict] = []
    for cand in sorted_cands:
        is_duplicate = False
        for existing in kept:
            sim = compute_hash_similarity(
                cand.get("image_hash", ""),
                existing.get("image_hash", ""),
            )
            if sim >= similarity_threshold:
                is_duplicate = True
                logger.debug(
                    "Dropping duplicate (similarity=%.4f): %s",
                    sim,
                    cand.get("image_path", ""),
                )
                break

        if not is_duplicate:
            kept.append(cand)

    logger.info(
        "Deduplication: %d candidates -> %d unique (threshold=%.2f).",
        len(candidates),
        len(kept),
        similarity_threshold,
    )
    return kept
