"""Sharpness — contrast-normalized Laplacian per tile, scored on the worst text.

The Laplacian is a second-derivative operator: it responds to places where
intensity changes abruptly, i.e. edges. Sharp text is almost entirely edges,
so the Laplacian carries a large share of the image's variation; blur smears
those edges and that share collapses.

Two things make a single global measurement wrong for phone captures:

* **Focus falloff.** The phone focuses on one part of the page and the far
  edge — often where a cheque's amount or signature sits — goes soft. A
  global variance averages the soft region away.
* **Content density.** A cheque is mostly blank, a claim form is dense. The
  same blur yields very different global variances on the two.

So the document is cut into tiles, blank tiles are dropped, each remaining
tile gets a sharpness ratio that does not depend on how dark its ink is, and
the score is taken from the blurriest tenth of the text.
"""

import cv2
import numpy as np

from scan_quality.scoring.base import (
    LABEL_UNKNOWN,
    CheckResult,
    ink_mask,
    label_for,
    log_linear_score,
    to_grayscale,
)

# --- Tuning constants -------------------------------------------------------

# Tile edge in pixels: 8 x 8 mm at the 8 px/mm working resolution. Small
# enough to localize a soft corner, large enough to hold several strokes.
TILE_PX = 64

# A tile counts as text when at least this share of its pixels is ink. Blank
# paper tiles carry only sensor noise, which has a *high* Laplacian ratio and
# would make a blank page look sharp.
MIN_INK_FRACTION = 0.03

# With at least this many text tiles the 10th percentile is stable; below it
# the median of the text tiles is used instead.
MIN_TILES_FOR_PERCENTILE = 10
WORST_TEXT_PERCENTILE = 10.0

# Score when no tile holds enough ink to measure. Neutral rather than 0: no
# text means nothing to judge focus on, and saying "hold still" would send the
# user after the wrong problem. A page with no readable ink is rejected by
# the contrast check instead, with the right instruction.
UNMEASURABLE_SCORE = 50.0

# Keeps the ratio finite on perfectly flat synthetic tiles.
VARIANCE_EPSILON = 1e-6

# Gaussian pre-smoothing (sigma, px) before the Laplacian. Sensor grain is
# pixel-to-pixel and the bare 3x3 Laplacian amplifies it ~20x, so on a
# grainy or washed-out photo the ratio would measure the noise, not the text.
# One pixel of smoothing removes most of that and leaves any blur worth
# rejecting (sigma >= ~1.5 px, i.e. 0.2 mm) clearly measurable.
PRESMOOTH_SIGMA_PX = 1.0

# Ratio that scores 0: blur of ~0.3 mm (sigma ~2.5 px), where small print
# merges into its neighbours.
BLUR_RATIO_FLOOR = 0.005

# Ratio that scores 100: crisp text at 8 px/mm, with no more than ~0.1 mm of
# blur. Higher values only come from over-sharpening halos.
SHARP_RATIO_CEILING = 0.10

CHECK_NAME = "sharpness"
RETAKE_MESSAGE = "Hold the phone still and parallel to the document."


def tile_sharpness(image: np.ndarray) -> np.ndarray:
    """Return the sharpness ratio of every text tile, in no particular order.

    The ratio is ``Var(Laplacian) / Var(intensity)`` on the lightly smoothed
    image: the share of the tile's variation that sits in fine detail.
    Dividing by the intensity variance makes it independent of ink darkness,
    so a pale pencil line and a black laser line of the same sharpness score
    alike.
    """
    gray = to_grayscale(image)
    ink = ink_mask(gray)
    intensity = cv2.GaussianBlur(gray.astype(np.float64), (0, 0), PRESMOOTH_SIGMA_PX)
    laplacian = cv2.Laplacian(intensity, cv2.CV_64F)

    ratios = []
    height, width = gray.shape
    for top in range(0, height - TILE_PX + 1, TILE_PX):
        for left in range(0, width - TILE_PX + 1, TILE_PX):
            window = (slice(top, top + TILE_PX), slice(left, left + TILE_PX))
            if ink[window].mean() < MIN_INK_FRACTION:
                continue
            ratios.append(laplacian[window].var() / (intensity[window].var() + VARIANCE_EPSILON))
    return np.asarray(ratios, dtype=np.float64)


def score_sharpness(document: np.ndarray) -> CheckResult:
    """Score how sharp the blurriest part of the text is, 0 (blurred) to 100 (crisp).

    Log ramp from :data:`BLUR_RATIO_FLOOR` (0) to :data:`SHARP_RATIO_CEILING`
    (100): the ratio falls by more than an order of magnitude as blur grows,
    and the visible difference between 0.005 and 0.015 is far larger than
    between 0.09 and 0.10.
    A document with no text tiles scores the neutral
    :data:`UNMEASURABLE_SCORE` with the label ``"unknown"``.
    """
    ratios = tile_sharpness(document)
    details = {"text_tiles": int(ratios.size), "tile_px": TILE_PX}

    if ratios.size == 0:
        return CheckResult(
            name=CHECK_NAME,
            score=UNMEASURABLE_SCORE,
            label=LABEL_UNKNOWN,
            raw_value=None,
            raw_metric="10th-percentile tile ratio Var(Laplacian)/Var(intensity)",
            details={**details, "reason": "no text found on the document"},
            retake_message=RETAKE_MESSAGE,
        )

    if ratios.size >= MIN_TILES_FOR_PERCENTILE:
        worst = float(np.percentile(ratios, WORST_TEXT_PERCENTILE))
    else:
        worst = float(np.median(ratios))
    score = log_linear_score(worst, BLUR_RATIO_FLOOR, SHARP_RATIO_CEILING)
    return CheckResult(
        name=CHECK_NAME,
        score=round(score, 2),
        label=label_for(score),
        raw_value=round(worst, 4),
        raw_metric="10th-percentile tile ratio Var(Laplacian)/Var(intensity)",
        details={**details, "median_ratio": round(float(np.median(ratios)), 4)},
        retake_message=RETAKE_MESSAGE,
    )
