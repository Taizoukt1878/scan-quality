"""Check 2 of 5 — lighting uniformity, via brightness spread across a grid.

A document photographed with a phone or fed through a worn scanner rarely gets
even illumination: one corner catches a lamp, the opposite edge falls into
shadow, or the page curls away from the platen. Locally that looks fine, but
globally it means a single binarization threshold cannot serve the whole page,
which is exactly what breaks downstream OCR.

The image is diced into a grid of cells; each cell contributes its mean
brightness. The standard deviation *across those cell means* is the raw
metric: it measures how much the page's illumination drifts from region to
region. Because it compares cell averages rather than individual pixels, it is
insensitive to the page's actual content — dense text and white margin have
similar means once averaged over a large cell — and responds mainly to
lighting gradients.
"""

import cv2
import numpy as np

from scan_quality.scoring.base import CheckResult, label_for, linear_score, to_grayscale

# --- Tuning constants -------------------------------------------------------

# Grid resolution. 8x8 = 64 samples: fine enough to catch a corner shadow or a
# one-sided gradient, coarse enough that each cell still averages over a large
# mix of ink and paper so text density does not masquerade as a lighting
# problem.
GRID_ROWS = 8
GRID_COLS = 8

# Brightness std-dev (0-255 scale) across cells that scores 100. Some spread is
# unavoidable and harmless — a page with a dark logo block or a photo will
# never be perfectly flat — so the ceiling is not pinned at 0.
UNIFORM_STD_FLOOR = 4.0

# Brightness std-dev across cells that scores 0. At this much drift the bright
# and dark regions of the page need genuinely different binarization
# thresholds; expect glare washout on one side or lost strokes on the other.
UNEVEN_STD_CEILING = 45.0

CHECK_NAME = "lighting_uniformity"


def cell_brightness_std(image: np.ndarray, rows: int = GRID_ROWS, cols: int = GRID_COLS) -> tuple[float, np.ndarray]:
    """Return the std-dev of per-cell mean brightness, and the cell-mean grid.

    The image is resized to an exact multiple of the grid so every cell holds
    the same pixel count — otherwise edge cells would be smaller and their
    means noisier, biasing the spread upward on some aspect ratios.
    """
    gray = to_grayscale(image)
    # Resize to a grid-aligned size. The interpolation choice is irrelevant for
    # means at this scale, but INTER_AREA keeps it an honest average.
    height = max(rows, (gray.shape[0] // rows) * rows)
    width = max(cols, (gray.shape[1] // cols) * cols)
    if (height, width) != gray.shape[:2]:
        gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)

    # Reshape into (rows, cell_h, cols, cell_w) and average each cell's pixels.
    cells = gray.reshape(rows, height // rows, cols, width // cols)
    cell_means = cells.mean(axis=(1, 3))
    return float(cell_means.std()), cell_means


def score_lighting_uniformity(image: np.ndarray) -> CheckResult:
    """Score how evenly lit a document image is, 0 (blotchy) to 100 (flat).

    Lower spread is better, so the ramp runs *backwards*:
    :data:`UNEVEN_STD_CEILING` maps to 0 and :data:`UNIFORM_STD_FLOOR` to 100,
    with linear interpolation between them. Linear is appropriate here because
    the metric is already in the same units as the pixel scale it describes —
    a 10-level drift is twice as bad as a 5-level one.
    """
    spread, cell_means = cell_brightness_std(image)
    score = linear_score(spread, UNEVEN_STD_CEILING, UNIFORM_STD_FLOOR)
    return CheckResult(
        name=CHECK_NAME,
        score=round(score, 2),
        label=label_for(score),
        raw_value=round(spread, 2),
        raw_metric="std-dev of per-cell mean brightness (0-255)",
        details={
            "grid": f"{GRID_ROWS}x{GRID_COLS}",
            "darkest_cell": round(float(cell_means.min()), 2),
            "brightest_cell": round(float(cell_means.max()), 2),
        },
    )
