"""Tests for the core pipeline modules."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import numpy as np
import tempfile
import os

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def create_test_image(path: str, width=640, height=480, color=(200, 200, 200)):
    """Create a simple test image."""
    img = np.full((height, width, 3), color, dtype=np.uint8)
    cv2.putText(img, "Test Demographics Income", (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.rectangle(img, (40, 150), (600, 400), (100, 100, 100), 2)
    # Add some lines to simulate table structure
    for y in range(180, 400, 30):
        cv2.line(img, (40, y), (600, y), (150, 150, 150), 1)
    cv2.imwrite(path, img)
    return path


@pytest.mark.skipif(not HAS_CV2, reason="OpenCV not installed")
class TestPreprocessFrame:
    def test_blur_score(self):
        from packages.core.pipeline.preprocess_frame import compute_blur_score
        # Sharp image (with edges/text) should have higher score
        sharp = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(sharp, (10, 10), (90, 90), (255, 255, 255), 2)
        cv2.putText(sharp, "SHARP", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Blurry image
        blurry = cv2.GaussianBlur(sharp, (31, 31), 10)

        sharp_score = compute_blur_score(sharp)
        blurry_score = compute_blur_score(blurry)
        assert sharp_score > blurry_score

    def test_preprocess_frame(self):
        from packages.core.pipeline.preprocess_frame import preprocess_frame
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            create_test_image(f.name)
            result = preprocess_frame(f.name)
            assert "blur_score" in result
            assert "is_sharp" in result
            assert "dimensions" in result
            os.unlink(f.name)


@pytest.mark.skipif(not HAS_CV2, reason="OpenCV not installed")
class TestRelevanceDetection:
    def test_keyword_detection(self):
        from packages.core.pipeline.detect_relevance import detect_relevance_by_keywords
        text = "The applicant demographics show income of $2000 and household size of 3"
        keywords = ["demographics", "income", "household", "eligibility"]
        score, matched = detect_relevance_by_keywords(text, keywords)
        assert score > 0
        assert "demographics" in matched
        assert "income" in matched
        assert "household" in matched
        assert "eligibility" not in matched

    def test_visual_structure(self):
        from packages.core.pipeline.detect_relevance import detect_visual_structure
        # Image with lines/structure should score higher
        structured = np.zeros((200, 200, 3), dtype=np.uint8)
        for y in range(20, 200, 20):
            cv2.line(structured, (10, y), (190, y), (255, 255, 255), 1)
        for x in range(20, 200, 40):
            cv2.line(structured, (x, 10), (x, 190), (255, 255, 255), 1)

        plain = np.full((200, 200, 3), 200, dtype=np.uint8)

        struct_score = detect_visual_structure(structured)
        plain_score = detect_visual_structure(plain)
        assert struct_score > plain_score


@pytest.mark.skipif(not HAS_CV2, reason="OpenCV not installed")
class TestDeduplication:
    def test_image_hash(self):
        from packages.core.pipeline.dedupe_candidates import compute_image_hash, compute_hash_similarity
        with tempfile.TemporaryDirectory() as tmpdir:
            # Two identical images should have similarity = 1.0
            path1 = create_test_image(os.path.join(tmpdir, "img1.png"))
            path2 = create_test_image(os.path.join(tmpdir, "img2.png"), color=(200, 200, 200))
            hash1 = compute_image_hash(path1)
            hash2 = compute_image_hash(path2)
            sim = compute_hash_similarity(hash1, hash2)
            assert sim == 1.0

            # Different image should have lower similarity
            path3 = create_test_image(os.path.join(tmpdir, "img3.png"), color=(50, 50, 200))
            hash3 = compute_image_hash(path3)
            sim2 = compute_hash_similarity(hash1, hash3)
            assert sim2 < 1.0


class TestClassification:
    def test_classify_demographics(self):
        from packages.core.pipeline.classify_screenshot import classify_screenshot
        result = classify_screenshot(
            extracted_text="Name: John Doe, Address: 123 Main St, Date of Birth: 01/01/1990",
            matched_keywords=["name", "address", "date of birth"],
            structure_score=0.3,
        )
        assert result["section_type"] == "demographics"
        assert result["confidence"] > 0

    def test_classify_income(self):
        from packages.core.pipeline.classify_screenshot import classify_screenshot
        result = classify_screenshot(
            extracted_text="Monthly income: $3,000. Employer: ACME Corp. Salary verification.",
            matched_keywords=["income", "salary"],
            structure_score=0.2,
        )
        assert result["section_type"] == "income"


class TestSynthesis:
    def test_synthesize_section(self):
        from packages.core.pipeline.synthesize_section import synthesize_section
        result = synthesize_section({
            "section_type": "demographics",
            "extracted_text": "Name: Jane Doe, DOB: 1985-03-15",
            "matched_keywords": ["name", "date of birth"],
            "confidence": 0.85,
        })
        assert "heading" in result
        assert "summary" in result
        assert "key_points" in result
        assert isinstance(result["key_points"], list)
