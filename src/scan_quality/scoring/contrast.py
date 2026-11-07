"""Check 3 of 5 — contrast, via RMS contrast (grayscale standard deviation).

RMS contrast is the standard deviation of pixel intensities across the whole
image. It answers "how far, on average, does a pixel sit from the page's mean
brightness". For a document that is really a question about ink separation: a
good scan is bimodal — a big cluster of near-white paper and a smaller cluster
of near-black ink — which puts a lot of mass far from the mean and yields a
high deviation. A washed-out scan (underexposed, faded thermal paper, a photo
through glare) compresses both clusters toward mid-gray, and the deviation
collapses.

Chosen over a plain histogram spread because it needs no bin count and no
percentile choices, and unlike max-minus-min it is not decided by a handful of
outlier pixels — a single dust speck cannot fake good contrast.
"""

import numpy as np

from scan_quality.scoring.base import CheckResult, label_for, linear_score, to_grayscale

# --- Tuning constants -------------------------------------------------------

# RMS contrast (0-255 scale) that scores 0. Below this the page is essentially
# uniform gray: ink and paper are no longer separable by threshold, and text is
# hard to read even for a human.
LOW_CONTRAST_FLOOR = 12.0

# RMS contrast that scores 100. A clean black-on-white scan of a dense page
# typically measures 60-90. The ceiling sits at the low end of that band rather
# than at the theoretical maximum (127.5, a pure half-black/half-white image)
# because sparse documents — a cheque, a short quote with wide margins — are
# mostly paper and can be perfectly legible at a lower deviation. Pinning the
# ceiling higher would penalize them for having margins.
HIGH_CONTRAST_CEILING = 65.0

CHECK_NAME = "contrast"


def rms_contrast(image: np.ndarray) -> float:
    """Return the RMS contrast: the std-dev of grayscale pixel intensities."""
    gray = to_grayscale(image)
    return float(np.asarray(gray, dtype=np.float64).std())


def score_contrast(image: np.ndarray) -> CheckResult:
    """Score ink/paper separation from 0 (washed out) to 100 (crisp bimodal).

    Linear ramp from :data:`LOW_CONTRAST_FLOOR` (0) to
    :data:`HIGH_CONTRAST_CEILING` (100). The metric is already expressed in
    intensity levels, so equal steps are equally meaningful and no log scaling
    is warranted.
    """
    contrast = rms_contrast(image)
    score = linear_score(contrast, LOW_CONTRAST_FLOOR, HIGH_CONTRAST_CEILING)
    return CheckResult(
        name=CHECK_NAME,
        score=round(score, 2),
        label=label_for(score),
        raw_value=round(contrast, 2),
        raw_metric="RMS contrast (std-dev of intensity, 0-255)",
    )
