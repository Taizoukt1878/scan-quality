"""Check 4 of 5 — skew, via the dominant text-line angle (Hough Transform).

A page fed crooked into a scanner, or photographed at an angle, produces text
baselines that are no longer horizontal. Even a couple of degrees hurts: OCR
engines segment text by projecting rows of pixels horizontally, and a tilted
baseline smears each row into its neighbours.

Method — Hough over *character baseline points*:

1. Binarize adaptively, so a page with a lighting gradient still yields clean
   glyph shapes across its whole width.
2. Find connected components and keep the ones shaped like characters,
   discarding speckle, rule lines and full-page borders.
3. Reduce each surviving glyph to a single pixel at the centre of its bottom
   edge. Every character on one line of text sits on that line's baseline, so
   these points are collinear.
4. Run the Hough Transform over that sparse point cloud. Each text line shows
   up as a peak; the median of the peak angles is the page skew.

The reduction in step 3 is what makes the estimate precise. Running Hough on
the text pixels directly — or on words dilated into solid bars, the obvious
shortcut — cannot resolve small angles: any line passing lengthwise through a
bar of height ``h`` and length ``L`` collects a full set of votes, so every
angle within ``±atan(h/L)`` ties. For ordinary body text that ambiguity cone
is well over a degree, swamping the range this check cares about. One-pixel
points have no thickness and no such cone.

The raw metric is the angle in degrees; the score penalizes its distance from
horizontal in either direction.
"""

import cv2
import numpy as np

from scan_quality.scoring.base import CheckResult, label_for, linear_score, resize_to_max_dim, to_grayscale

# --- Tuning constants -------------------------------------------------------

# Hough's cost grows with pixel count and its angular resolution does not
# improve much past this size, so cap the working image for predictable
# runtime on large scans. All pixel-denominated constants below are relative
# to this working size.
WORKING_MAX_DIM = 1600

# Adaptive (rather than Otsu) binarization: a single global threshold loses
# the shadowed half of an unevenly lit page, which then contributes no
# baseline points and biases the estimate toward whatever survives. Block size
# must be odd and comfortably larger than one character; C is subtracted from
# the local mean to keep faint paper texture out of the foreground.
ADAPTIVE_BLOCK_SIZE = 31
ADAPTIVE_C = 10

# Which connected components count as characters. Heights and widths are
# fractions of the working image so they hold at any input resolution.
GLYPH_MIN_HEIGHT_PX = 4        # below this it is speckle or JPEG noise
GLYPH_MAX_HEIGHT_FRAC = 0.10   # taller than a tenth of the page = border/figure
GLYPH_MAX_WIDTH_FRAC = 0.10    # wider than a tenth of the page = rule line
GLYPH_MIN_AREA_PX = 8          # discards isolated dust specks

# Below this many baseline points the page has too little text to fit lines to
# (a near-blank cheque stub, a photo of an empty page).
MIN_BASELINE_POINTS = 20

# Hough parameters over the point cloud. 1440 theta steps = 0.125-degree
# resolution, which is finer than this check needs to distinguish "straight"
# from "slightly crooked" and cheap on a sparse image. A peak needs
# HOUGH_MIN_VOTES collinear characters to count as a text line.
HOUGH_RHO_PX = 1
HOUGH_THETA_STEPS = 1440
HOUGH_MIN_VOTES = 6

# Peaks steeper than this are not text baselines — they are table verticals or
# accidental alignments down a column of text — and would corrupt the median.
# Also bounds what this check can see: it detects the small-angle skew that
# scanning introduces, not a 90-degree rotation.
MAX_ANALYZED_ANGLE_DEG = 20.0

# Below this many usable peaks the median is not trustworthy.
MIN_CANDIDATE_LINES = 3

# Score when skew cannot be measured at all. Deliberately neutral rather than
# 0 or 100: the page is not known to be crooked, but nothing confirmed it is
# straight either, so it should neither reward nor punish the overall average.
UNDETECTABLE_SCORE = 50.0

# |angle| that scores 100. Sub-degree skew is invisible and harmless — most
# OCR deskew stages will not even engage below it.
PERFECT_SKEW_DEG = 0.5

# |angle| that scores 0. Past roughly this much tilt, horizontal projection
# profiles overlap between adjacent baselines and OCR accuracy drops sharply
# unless the page is deskewed first.
MAX_TOLERATED_SKEW_DEG = 5.0

CHECK_NAME = "skew"


def baseline_points(image: np.ndarray) -> np.ndarray:
    """Reduce ``image`` to a mask holding one pixel per detected character.

    Each set pixel sits at the centre of a glyph's bottom edge, i.e. on that
    character's baseline. See the module docstring for why this reduction
    matters.
    """
    gray = resize_to_max_dim(to_grayscale(image), WORKING_MAX_DIM)
    # THRESH_BINARY_INV so ink becomes the foreground that component labelling
    # looks for.
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        ADAPTIVE_BLOCK_SIZE, ADAPTIVE_C,
    )

    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    height, width = binary.shape
    points = np.zeros((height, width), dtype=np.uint8)
    if count <= 1:  # label 0 is the background, so this means "no ink at all"
        return points

    # Row 0 is the background component; drop it before filtering.
    stats = stats[1:]
    lefts = stats[:, cv2.CC_STAT_LEFT]
    tops = stats[:, cv2.CC_STAT_TOP]
    widths = stats[:, cv2.CC_STAT_WIDTH]
    heights = stats[:, cv2.CC_STAT_HEIGHT]
    areas = stats[:, cv2.CC_STAT_AREA]

    is_glyph = (
        (heights >= GLYPH_MIN_HEIGHT_PX)
        & (heights <= GLYPH_MAX_HEIGHT_FRAC * height)
        & (widths <= GLYPH_MAX_WIDTH_FRAC * width)
        & (areas >= GLYPH_MIN_AREA_PX)
    )
    if not is_glyph.any():
        return points

    rows = np.clip(tops[is_glyph] + heights[is_glyph] - 1, 0, height - 1)
    cols = np.clip(lefts[is_glyph] + widths[is_glyph] // 2, 0, width - 1)
    points[rows, cols] = 255
    return points


def detect_skew_angle(image: np.ndarray) -> tuple[float | None, int, int]:
    """Estimate page skew; return ``(angle_degrees, peak_count, point_count)``.

    Positive angles mean the text rises to the right. ``angle_degrees`` is
    ``None`` when the page yields too few baseline points or too few Hough
    peaks for a trustworthy estimate.

    The median (not the mean) of the peak angles is used: a stray peak from a
    signature stroke, a fold shadow or a stamp would drag a mean off the true
    baseline angle, whereas the median ignores it.
    """
    points = baseline_points(image)
    point_count = int(np.count_nonzero(points))
    if point_count < MIN_BASELINE_POINTS:
        return None, 0, point_count

    lines = cv2.HoughLines(
        points, HOUGH_RHO_PX, np.pi / HOUGH_THETA_STEPS, HOUGH_MIN_VOTES
    )
    if lines is None:
        return None, 0, point_count

    # OpenCV parameterises a line by the angle theta of its *normal*, measured
    # from the x-axis: a perfectly horizontal line has theta = 90 degrees.
    # Skew is therefore the signed departure of theta from 90, negated so that
    # "text rising to the right" comes out positive in ordinary maths
    # orientation despite image rows growing downward.
    angles = [
        angle
        for angle in (90.0 - np.degrees(theta) for _, theta in lines[:, 0])
        if abs(angle) <= MAX_ANALYZED_ANGLE_DEG
    ]
    if len(angles) < MIN_CANDIDATE_LINES:
        return None, len(angles), point_count
    return float(np.median(angles)), len(angles), point_count


def score_skew(image: np.ndarray) -> CheckResult:
    """Score page straightness from 0 (badly tilted) to 100 (square).

    Only the magnitude of the angle matters — a page rotated 3 degrees
    clockwise is exactly as bad as 3 degrees counter-clockwise — so the score
    ramps linearly on ``abs(angle)`` from :data:`PERFECT_SKEW_DEG` (100) down
    to :data:`MAX_TOLERATED_SKEW_DEG` (0). The signed angle is still reported
    as ``raw_value``, because a caller wanting to auto-deskew needs the
    direction.

    When no reliable baselines exist, the score is the neutral
    :data:`UNDETECTABLE_SCORE` and the label is ``"unknown"``.
    """
    angle, line_count, point_count = detect_skew_angle(image)

    if angle is None:
        reason = (
            f"only {point_count} character baseline points found, need "
            f"{MIN_BASELINE_POINTS}"
            if point_count < MIN_BASELINE_POINTS
            else (
                f"found {line_count} candidate text lines, need "
                f"{MIN_CANDIDATE_LINES} for a reliable estimate"
            )
        )
        return CheckResult(
            name=CHECK_NAME,
            score=UNDETECTABLE_SCORE,
            label="unknown",
            raw_value=None,
            raw_metric="dominant text-line angle (degrees from horizontal)",
            details={
                "text_lines_detected": line_count,
                "baseline_points": point_count,
                "reason": reason,
            },
        )

    score = linear_score(abs(angle), MAX_TOLERATED_SKEW_DEG, PERFECT_SKEW_DEG)
    return CheckResult(
        name=CHECK_NAME,
        score=round(score, 2),
        label=label_for(score),
        raw_value=round(angle, 3),
        raw_metric="dominant text-line angle (degrees from horizontal)",
        details={"text_lines_detected": line_count, "baseline_points": point_count},
    )
