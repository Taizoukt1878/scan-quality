"""Glare — the largest saturated spot on the document.

Glossy paper, laminated cards and documents photographed under a ceiling
light reflect the light source straight into the lens. Where that happens the
sensor saturates in every channel and whatever was printed underneath is
gone: unlike a shadow, glare cannot be corrected afterwards. One spot over a
cheque's amount ruins the capture however good everything else is, so this
check is strict and judges the single largest spot rather than an average.

Strong overexposure saturates the paper the same way and is caught too — it
erases faint ink just as thoroughly.
"""

import cv2
import numpy as np

from scan_quality.scoring.base import CheckResult, label_for, linear_score, min_channel

# --- Tuning constants -------------------------------------------------------

# A pixel is saturated when even its darkest channel is at least this bright.
# Requiring all three channels separates white glare from a merely bright
# coloured area, and real paper under normal exposure stays below it.
SATURATION_LEVEL = 250

# Opening kernel that removes isolated saturated pixels (sensor hot pixels,
# JPEG ringing) before blobs are measured.
SPECK_OPEN_PX = 3

# Blobs smaller than this cannot hide a character at 8 px/mm and are ignored.
MIN_BLOB_PX = 20

# Largest-blob share of the document that scores 100: 0.05 % of a cheque is
# ~450 px, about 2.5 x 2.5 mm — smaller than one handwritten digit.
HARMLESS_GLARE_FRAC = 0.0005

# Largest-blob share that scores 0: 1 % of a cheque is ~9 000 px, about
# 12 x 12 mm — enough to hide the amount in figures.
FATAL_GLARE_FRAC = 0.01

CHECK_NAME = "glare"
RETAKE_MESSAGE = "Tilt the phone slightly to remove the reflection."


def largest_glare_blob(document: np.ndarray) -> tuple[int, int]:
    """Return ``(largest_blob_px, blob_count)`` of saturated regions."""
    saturated = (min_channel(document) >= SATURATION_LEVEL).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (SPECK_OPEN_PX, SPECK_OPEN_PX))
    saturated = cv2.morphologyEx(saturated, cv2.MORPH_OPEN, kernel)

    count, _, stats, _ = cv2.connectedComponentsWithStats(saturated, connectivity=8)
    # Row 0 is the unsaturated background component.
    areas = stats[1:count, cv2.CC_STAT_AREA]
    areas = areas[areas >= MIN_BLOB_PX]
    if areas.size == 0:
        return 0, 0
    return int(areas.max()), int(areas.size)


def score_glare(document: np.ndarray) -> CheckResult:
    """Score from 0 (a spot large enough to hide a field) to 100 (no glare).

    The raw value is the largest blob's share of the document area, mapped
    with a reversed linear ramp from :data:`FATAL_GLARE_FRAC` (0) to
    :data:`HARMLESS_GLARE_FRAC` (100). The largest blob rather than the total,
    because one solid spot hides a word while scattered specks hide nothing.
    """
    largest, count = largest_glare_blob(document)
    fraction = largest / float(document.shape[0] * document.shape[1])
    score = linear_score(fraction, FATAL_GLARE_FRAC, HARMLESS_GLARE_FRAC)
    return CheckResult(
        name=CHECK_NAME,
        score=round(score, 2),
        label=label_for(score),
        raw_value=round(fraction * 100.0, 4),
        raw_metric="largest saturated blob (% of document area)",
        details={"largest_blob_px": largest, "blob_count": count},
        retake_message=RETAKE_MESSAGE,
    )
