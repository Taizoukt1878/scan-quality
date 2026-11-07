"""Shadows — how much less light the darkest part of the paper receives.

The user's hand, arm or phone casts shadows on a document photographed from
above. The question is not how bright the page is overall but how unevenly
lit it is, and only the *paper* says that: ink is dark under any light.

So the ink is removed first (see :func:`~scan_quality.scoring.base.estimate_background`),
leaving an estimate of the light falling on the paper at every pixel. The raw
metric is the ratio of its dark end to its bright end. A ratio rather than a
difference, so a dim but evenly lit room is not penalized.

Anchors are deliberately lenient. A soft gradient is easy to divide out
downstream; what a deep shadow really costs is ink contrast, and the contrast
check measures that directly in the worst region. This check mainly exists to
tell the user *why* — "move the shadow" is more actionable than "low contrast".
"""

import numpy as np

from scan_quality.scoring.base import CheckResult, estimate_background, label_for, linear_score, to_grayscale

# --- Tuning constants -------------------------------------------------------

# Percentiles of the background used as its dark and bright ends. Not the
# minimum and maximum, so a dark logo block or a few pixels at the warped
# document edge cannot decide the result.
DARK_PERCENTILE = 5.0
BRIGHT_PERCENTILE = 95.0

# Shading ratio that scores 0: the shadowed part of the page gets 40 % of the
# light the brightest part gets — a hard arm shadow.
SHADED_RATIO_FLOOR = 0.40

# Shading ratio that scores 100. A soft window gradient lands around here and
# is harmless.
EVEN_RATIO_CEILING = 0.80

CHECK_NAME = "shadows"
RETAKE_MESSAGE = "Move to even light and avoid casting a shadow on the document."


def shading_ratio(document: np.ndarray) -> tuple[float, float, float]:
    """Return ``(ratio, dark_end, bright_end)`` of the paper's brightness."""
    background = estimate_background(to_grayscale(document)).astype(np.float64)
    dark = float(np.percentile(background, DARK_PERCENTILE))
    bright = float(np.percentile(background, BRIGHT_PERCENTILE))
    # An all-black document has no light to compare; call it fully shaded.
    ratio = dark / bright if bright > 0 else 0.0
    return ratio, dark, bright


def score_shadows(document: np.ndarray) -> CheckResult:
    """Score lighting evenness from 0 (hard shadow) to 100 (even light).

    Linear ramp from :data:`SHADED_RATIO_FLOOR` (0) to
    :data:`EVEN_RATIO_CEILING` (100).
    """
    ratio, dark, bright = shading_ratio(document)
    score = linear_score(ratio, SHADED_RATIO_FLOOR, EVEN_RATIO_CEILING)
    return CheckResult(
        name=CHECK_NAME,
        score=round(score, 2),
        label=label_for(score),
        raw_value=round(ratio, 4),
        raw_metric="paper brightness ratio p5/p95 (0-1)",
        details={"dark_paper": round(dark, 2), "bright_paper": round(bright, 2)},
        retake_message=RETAKE_MESSAGE,
    )
