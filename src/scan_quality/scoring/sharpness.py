"""Check 1 of 5 — sharpness / blur, via variance of the Laplacian.

The Laplacian is a second-derivative operator: it responds to places where
intensity changes abruptly, i.e. edges. A sharp scan of a document is almost
entirely edges (every glyph stroke has two of them), so the Laplacian response
is large and widely spread, giving a high variance. Blurring smears those
transitions across many pixels, shrinking the response toward zero everywhere
and collapsing the variance.

So the raw metric is "how much edge energy does this image contain", and the
score answers "is that consistent with an in-focus document scan".
"""

import cv2
import numpy as np

from scan_quality.scoring.base import (
    CheckResult,
    label_for,
    log_linear_score,
    resize_to_max_dim,
    to_grayscale,
)

# --- Tuning constants -------------------------------------------------------

# Laplacian variance scales with resolution: the same physical blur measured on
# a 4000px scan yields a far larger variance than on a 1000px one. Every image
# is therefore resampled to this longest-side budget before measuring, so the
# thresholds below mean the same thing for every upload.
WORKING_MAX_DIM = 1500

# Raw variance that scores 0. Below this the page reads as visibly out of
# focus or motion-blurred — glyph edges have been smeared into their
# background and OCR starts dropping whole words.
BLUR_VARIANCE_FLOOR = 15.0

# Raw variance that scores 100. A crisp 300-DPI flatbed scan of printed text
# lands well above this; pushing the ceiling higher would only reward sensor
# noise and over-sharpening halos, which are not quality.
SHARP_VARIANCE_CEILING = 600.0

CHECK_NAME = "sharpness"


def laplacian_variance(image: np.ndarray) -> float:
    """Return the raw Laplacian variance of ``image`` at the working size.

    Exposed separately from :func:`score_sharpness` so the unnormalized
    measurement can be inspected or re-tuned against real scans.
    """
    gray = to_grayscale(image)
    gray = resize_to_max_dim(gray, WORKING_MAX_DIM)
    # CV_64F keeps the negative half of the Laplacian response; an unsigned
    # output type would clip it to zero and roughly halve the variance.
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def score_sharpness(image: np.ndarray) -> CheckResult:
    """Score how in-focus a document image is, from 0 (blurred) to 100 (crisp).

    The raw Laplacian variance is mapped onto the score with a *logarithmic*
    ramp between :data:`BLUR_VARIANCE_FLOOR` and
    :data:`SHARP_VARIANCE_CEILING`. Log rather than linear because the metric
    spans orders of magnitude — the perceptual gap between variance 20 and 60
    is much larger than between 500 and 540, and a linear ramp would squash
    every ordinary scan into the bottom of the range.
    """
    variance = laplacian_variance(image)
    score = log_linear_score(variance, BLUR_VARIANCE_FLOOR, SHARP_VARIANCE_CEILING)
    return CheckResult(
        name=CHECK_NAME,
        score=round(score, 2),
        label=label_for(score),
        raw_value=round(variance, 2),
        raw_metric="Laplacian variance",
        details={"working_max_dim": WORKING_MAX_DIM},
    )
