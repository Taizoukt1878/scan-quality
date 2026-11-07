"""Shared primitives for the capture-quality checks.

Every pixel check module in this package follows the same contract:

    score_<check>(document: np.ndarray) -> CheckResult

The input is the *rectified document* produced by
:mod:`scan_quality.scoring.document` — the page cut out of the photo and
flattened to a known physical size — as a 3-channel BGR or single-channel
grayscale array. The output is a :class:`CheckResult` carrying both the
normalized 0-100 score and the raw physical measurement it came from.

Checks are pure functions of the pixel data: no I/O, no FastAPI, no globals.
That keeps them trivially testable (feed an array, assert on the score) and
lets the pipeline run them in any order.
"""

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

# --- Label thresholds -------------------------------------------------------
# The same score->word mapping is applied to every check and to the overall
# score, so a "good" sharpness and a "good" contrast mean the same thing to
# whoever reads the JSON. Boundaries are inclusive lower bounds.
#
# The cut points are set so that a score of 50 is the default accept line: below it
# the capture has a defect a human would notice and should be retaken.
LABEL_EXCELLENT_MIN = 85.0  # visually indistinguishable from a clean scan
LABEL_GOOD_MIN = 70.0       # minor imperfection, OCR unaffected in practice
LABEL_FAIR_MIN = 50.0       # usable but degraded; default accept threshold
# Anything below LABEL_FAIR_MIN is "poor" -> retake.

LABEL_UNKNOWN = "unknown"   # used when a check cannot measure anything at all


@dataclass
class CheckResult:
    """Outcome of a single quality check.

    Attributes:
        name: Stable machine-readable identifier, e.g. ``"sharpness"``.
        score: Normalized quality on a 0-100 scale, where 100 is best. This is
            the only field that participates in the overall average.
        label: Human-readable bucket for ``score`` ("excellent" ... "poor").
        raw_value: The underlying physical measurement before normalization
            (tile sharpness ratio, px/mm, ...). Kept in the response so
            you can re-tune the module constants against real-world scans
            without re-instrumenting the service. ``None`` when the check could
            not measure the image at all.
        raw_metric: Human-readable name and unit of ``raw_value``.
        details: Optional extra diagnostics specific to one check (tile counts,
            corner positions, ...). Never used in scoring.
        retake_message: The instruction to show the user if this check is the
            one that rejects the capture.
    """

    name: str
    score: float
    label: str
    raw_value: float | None
    raw_metric: str
    details: dict[str, Any] = field(default_factory=dict)
    retake_message: str | None = None


def label_for(score: float) -> str:
    """Map a 0-100 score onto the shared quality vocabulary.

    Uses the LABEL_* constants above so that every check and the overall
    score speak the same language.
    """
    if score >= LABEL_EXCELLENT_MIN:
        return "excellent"
    if score >= LABEL_GOOD_MIN:
        return "good"
    if score >= LABEL_FAIR_MIN:
        return "fair"
    return "poor"


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    """Constrain ``value`` to the inclusive range [low, high]."""
    return float(max(low, min(high, value)))


def linear_score(value: float, zero_at: float, hundred_at: float) -> float:
    """Linearly map a raw measurement onto 0-100, clamped at both ends.

    ``zero_at`` is the raw value that earns a score of 0 and ``hundred_at`` the
    raw value that earns 100. The mapping is deliberately direction-agnostic:
    when ``zero_at > hundred_at`` the scale inverts, which is what "lower is
    better" metrics (glare area) need.

    Values beyond either anchor are clamped rather than extrapolated, so a
    wildly sharp image cannot earn 130 and drag an average upward.
    """
    span = hundred_at - zero_at
    if span == 0:
        raise ValueError("zero_at and hundred_at must differ")
    return clamp(100.0 * (value - zero_at) / span)


def log_linear_score(value: float, zero_at: float, hundred_at: float) -> float:
    """Like :func:`linear_score` but interpolates in log10 space.

    Some raw metrics (notably Laplacian variance) span orders of magnitude:
    the difference between variance 10 and 100 is a much bigger perceptual
    jump than between 900 and 990. Interpolating on the logarithm spreads the
    score evenly over that dynamic range instead of squashing every ordinary
    image into the bottom few points.

    Both anchors must be strictly positive. Non-positive measurements are
    treated as the worst case and score 0.
    """
    if zero_at <= 0 or hundred_at <= 0:
        raise ValueError("log_linear_score anchors must be > 0")
    if value <= 0:
        # A flat image (zero variance) is the degenerate worst case.
        return 0.0 if hundred_at > zero_at else 100.0
    return linear_score(
        float(np.log10(value)), float(np.log10(zero_at)), float(np.log10(hundred_at))
    )


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Return a single-channel uint8 view of ``image``.

    Accepts BGR, BGRA or already-grayscale input so callers never have to know
    what the loader produced. Grayscale input is passed through untouched.
    """
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Unsupported image shape for grayscale conversion: {image.shape}")
    return gray if gray.dtype == np.uint8 else cv2.convertScaleAbs(gray)


def min_channel(image: np.ndarray) -> np.ndarray:
    """Return the per-pixel minimum over the colour channels, as uint8.

    Paper is bright in every channel, but any coloured ink is dark in at
    least one: red stamps in green and blue, blue ballpoint in red. Taking the
    minimum keeps every ink dark, where a luminance conversion would let a
    pale red stamp fade almost into the paper. Grayscale input is returned
    unchanged.
    """
    if image.ndim == 2:
        return image
    return image[:, :, :3].min(axis=2)


# Ink removal for background estimation: a closing (dilate then erode) with a
# kernel wider than any pen or print stroke fills ink in with the surrounding
# paper. 15 px is ~2 mm at the 8 px/mm working resolution.
BACKGROUND_CLOSE_PX = 15
# Smoothing after ink removal, so what remains is the slowly varying light
# falling on the paper rather than paper texture: a median over ~6 mm. A
# median rather than a Gaussian because it keeps a hard shadow edge sharp; a
# blurred edge would leave a band of mis-corrected paper that reads as faint
# ink. Must be odd.
BACKGROUND_MEDIAN_PX = 51


def estimate_background(gray: np.ndarray) -> np.ndarray:
    """Estimate the paper's brightness at every pixel, with the ink removed.

    Used to flat-field correct the page (divide by it) and, on its own, to
    measure shadows. Large dark regions — a black logo block, a photo — are
    wider than the closing kernel and survive as dark background.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (BACKGROUND_CLOSE_PX, BACKGROUND_CLOSE_PX))
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    return cv2.medianBlur(closed, BACKGROUND_MEDIAN_PX)


# Adaptive (rather than Otsu) binarization: a single global threshold loses
# the shadowed half of an unevenly lit page. Block size must be odd and
# comfortably larger than one character stroke; C is subtracted from the local
# mean to keep faint paper texture out of the foreground.
INK_BLOCK_SIZE = 31
INK_C = 10
# Opening applied to the mask. Sensor noise crosses the threshold as isolated
# single pixels, while every real stroke is at least ~2 px (0.25 mm) wide, so
# a 2 x 2 opening keeps the strokes and drops the noise. Without it, blank
# paper on a grainy photo would count as faint ink.
INK_OPEN_PX = 2


def ink_mask(gray: np.ndarray) -> np.ndarray:
    """Return a boolean mask that is True on ink pixels."""
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        INK_BLOCK_SIZE, INK_C,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (INK_OPEN_PX, INK_OPEN_PX))
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel) > 0


def resize_to_max_dim(image: np.ndarray, max_dim: int) -> np.ndarray:
    """Downscale ``image`` so its longest side is at most ``max_dim`` pixels.

    Several raw metrics are resolution-dependent — the same physical blur
    produces a much larger Laplacian variance on a 4000px scan than on a
    1000px one. Normalizing the working size first makes scores comparable
    across uploads of different DPI. Images already at or below the limit are
    returned unchanged (never upscaled, which would invent detail).
    """
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_dim:
        return image
    scale = max_dim / float(longest)
    # INTER_AREA is the correct filter for shrinking: it averages the pixels
    # being merged instead of point-sampling, so it does not fabricate edges
    # that would inflate a sharpness measurement.
    return cv2.resize(
        image, (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
