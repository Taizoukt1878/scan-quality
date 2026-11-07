"""Contrast — how clearly ink stands out from the paper around it.

What decides whether ink can be read is not how much of the page is ink —
the old RMS contrast penalized cheques for being mostly blank — but whether
each ink pixel can be told apart from the paper around it. Two conditions,
both required:

* **Separation** — the ink-to-paper gap divided by the paper's own noise
  (sensor grain, JPEG artefacts, paper texture). A signal-to-noise ratio:
  below a few sigma, binarization breaks strokes apart.
* **Depth** — the gap as a share of the paper's brightness (Weber contrast).
  A very faint ink on a very clean photo can be statistically separable and
  still be lost to the phone's JPEG compression or to the downstream
  binarizer's heuristics. Depth puts a floor under that.

Three preparation steps make that gap mean the same thing everywhere:

1. **Min-channel image.** Every coloured ink stays dark (see
   :func:`~scan_quality.scoring.base.min_channel`), so a pale red stamp is
   judged on its real contrast rather than its faint luminance.
2. **Flat-field correction.** Dividing by the estimated paper background
   removes shadows, so they are not counted twice; the noise they amplify is
   still counted, which is the real cost of a deep shadow.
3. **Worst cell.** The page is split into a grid and the lowest cell decides,
   so a faded corner or a shadowed strip is not averaged away.

Why the paper's noise and not a pooled ink-and-paper deviation: the ink class
is dominated by anti-aliased stroke edges, whose spread grows with contrast
itself, so a pooled ratio barely moves when the page is washed out.
"""

import numpy as np

from scan_quality.scoring.base import (
    CheckResult,
    estimate_background,
    ink_mask,
    label_for,
    linear_score,
    min_channel,
)

# --- Tuning constants -------------------------------------------------------

# Grid the document is split into; the worst cell decides.
GRID_ROWS = 4
GRID_COLS = 4

# A cell needs this many ink pixels for its ink level to be trusted.
MIN_INK_PX = 200

# Floor on the paper noise. An 8-bit JPEG never delivers a cleaner paper than
# this, and without it a noiseless synthetic page would divide by zero.
NOISE_FLOOR = 2.0

# MAD -> standard deviation for normally distributed noise.
MAD_TO_SIGMA = 1.4826

# Ink-to-paper gap, in paper-noise units, that scores 0. At 2 sigma a
# threshold half-way between ink and paper misclassifies ~16 % of pixels on
# each side, and strokes break up.
LOW_SEPARATION_FLOOR = 2.0

# Gap that scores 100: past 6 sigma binarization is effectively error-free.
HIGH_SEPARATION_CEILING = 6.0

# Ink depth, 1 - ink/paper, that scores 0: ink only 8 % darker than the
# paper — a ghost of a stamp, a nearly dry pen.
SHALLOW_INK_FLOOR = 0.08

# Ink depth that scores 100: ink a quarter darker than the paper, which even
# pencil and faded thermal print reach on a usable capture.
DEEP_INK_CEILING = 0.25

CHECK_NAME = "contrast"
RETAKE_MESSAGE = "Find more light; the text is too faint to read."


def flat_field(document: np.ndarray) -> np.ndarray:
    """Return the min-channel image divided by its background, on 0-255."""
    channel = min_channel(document)
    background = estimate_background(channel).astype(np.float64)
    corrected = 255.0 * channel.astype(np.float64) / np.maximum(background, 1.0)
    return np.clip(corrected, 0, 255).astype(np.uint8)


def cell_measurements(document: np.ndarray) -> list[tuple[float, float]]:
    """Return ``(separation, depth)`` for every grid cell that holds ink."""
    corrected = flat_field(document)
    ink = ink_mask(corrected)
    height, width = corrected.shape
    row_edges = np.linspace(0, height, GRID_ROWS + 1, dtype=int)
    col_edges = np.linspace(0, width, GRID_COLS + 1, dtype=int)

    measurements = []
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            window = (slice(row_edges[r], row_edges[r + 1]), slice(col_edges[c], col_edges[c + 1]))
            cell, cell_ink = corrected[window].astype(np.float64), ink[window]
            if cell_ink.sum() < MIN_INK_PX or (~cell_ink).sum() < MIN_INK_PX:
                continue
            paper, inked = cell[~cell_ink], cell[cell_ink]
            paper_level = np.median(paper)
            # Median absolute deviation: robust to the odd ink pixel the mask
            # missed, which a standard deviation would turn into "noise".
            noise = max(MAD_TO_SIGMA * np.median(np.abs(paper - paper_level)), NOISE_FLOOR)
            gap = paper_level - np.median(inked)
            measurements.append((float(gap / noise), float(gap / max(paper_level, 1.0))))
    return measurements


def score_contrast(document: np.ndarray) -> CheckResult:
    """Score ink legibility from 0 (lost) to 100 (clean, deep ink).

    Each condition gets a linear ramp on its worst cell — separation from
    :data:`LOW_SEPARATION_FLOOR` to :data:`HIGH_SEPARATION_CEILING`, depth
    from :data:`SHALLOW_INK_FLOOR` to :data:`DEEP_INK_CEILING` — and the
    score is the lower of the two. ``raw_value`` is the worst separation; the
    worst depth is in ``details``. A document with no measurable ink scores 0.
    """
    measurements = cell_measurements(document)
    metric = "worst-cell ink/paper gap in paper-noise units"

    if not measurements:
        return CheckResult(
            name=CHECK_NAME,
            score=0.0,
            label=label_for(0.0),
            raw_value=None,
            raw_metric=metric,
            details={"cells_measured": 0, "reason": "no ink found on the document"},
            retake_message=RETAKE_MESSAGE,
        )

    worst_separation = min(separation for separation, _ in measurements)
    worst_depth = min(depth for _, depth in measurements)
    score = min(
        linear_score(worst_separation, LOW_SEPARATION_FLOOR, HIGH_SEPARATION_CEILING),
        linear_score(worst_depth, SHALLOW_INK_FLOOR, DEEP_INK_CEILING),
    )
    return CheckResult(
        name=CHECK_NAME,
        score=round(score, 2),
        label=label_for(score),
        raw_value=round(worst_separation, 3),
        raw_metric=metric,
        details={
            "worst_ink_depth": round(worst_depth, 4),
            "cells_measured": len(measurements),
            "grid": f"{GRID_ROWS}x{GRID_COLS}",
        },
        retake_message=RETAKE_MESSAGE,
    )
