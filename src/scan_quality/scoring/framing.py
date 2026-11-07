"""Framing and resolution — is the whole document there, with enough pixels?

Rotation and perspective are not scored: the rectification step removes them
and the downstream OCR deskews whatever is left. What a phone capture *can*
lose for good is framing — part of the document outside the photo — and
resolution — the document so far away that the detail was never recorded.

Resolution is measured on the worst side of the detected document, in pixels
per millimetre of real paper. A capture shot at an angle has fewer pixels on
its far side; taking the minimum is how perspective is penalized, in the one
way that matters.
"""

import numpy as np

from scan_quality.scoring.base import CheckResult, label_for, linear_score
from scan_quality.scoring.document import DocumentDetection, side_resolutions_px_per_mm

# --- Tuning constants -------------------------------------------------------

# Each corner must sit at least this share of the frame away from its edge.
# A corner closer than that is almost always one the user cropped off, with
# the detector snapping to the frame border instead.
EDGE_MARGIN_FRAC = 0.01

# Worst-side resolution that scores 0: ~100 DPI, where handwriting and small
# print lose their stroke detail.
MIN_PX_PER_MM = 4.0

# Worst-side resolution that scores 100: ~200 DPI, the working resolution the
# document is rectified to.
TARGET_PX_PER_MM = 8.0

CHECK_NAME = "framing"
# Not found has two usual causes the photo cannot tell apart: the page fills
# or overflows the frame, or it lies on a surface as light as itself.
NOT_FOUND_MESSAGE = "Fit the whole document in the frame, on a darker plain surface."
CUT_OFF_MESSAGE = "Fit the whole document inside the frame."
TOO_FAR_MESSAGE = "Move closer to the document."


def rotation_degrees(detection: DocumentDetection) -> float:
    """Angle of the document's top edge; positive when it rises to the right."""
    top_left, top_right = detection.corners[0], detection.corners[1]
    # Negated because image rows grow downward.
    return float(-np.degrees(np.arctan2(top_right[1] - top_left[1], top_right[0] - top_left[0])))


def touches_frame_edge(detection: DocumentDetection) -> bool:
    """True when any corner sits within :data:`EDGE_MARGIN_FRAC` of the frame edge."""
    width, height = detection.frame_size
    margin_x, margin_y = EDGE_MARGIN_FRAC * width, EDGE_MARGIN_FRAC * height
    xs, ys = detection.corners[:, 0], detection.corners[:, 1]
    return bool(
        (xs < margin_x).any() or (xs > width - 1 - margin_x).any()
        or (ys < margin_y).any() or (ys > height - 1 - margin_y).any()
    )


def score_framing(detection: DocumentDetection | None, doc_type: str) -> CheckResult:
    """Score from 0 (cut off, or too far away) to 100 (whole document, full detail).

    A document that was not found or touches the frame edge scores 0
    whatever its resolution. Otherwise the worst side's px/mm is mapped with
    a linear ramp from :data:`MIN_PX_PER_MM` (0) to :data:`TARGET_PX_PER_MM`
    (100).
    """
    metric = "worst-side resolution (px per mm of document)"

    if detection is None:
        return CheckResult(
            name=CHECK_NAME,
            score=0.0,
            label=label_for(0.0),
            raw_value=None,
            raw_metric=metric,
            details={"document_found": False, "doc_type": doc_type},
            retake_message=NOT_FOUND_MESSAGE,
        )

    resolution = float(side_resolutions_px_per_mm(detection, doc_type).min())
    cut_off = not detection.whole_frame and touches_frame_edge(detection)
    score = 0.0 if cut_off else linear_score(resolution, MIN_PX_PER_MM, TARGET_PX_PER_MM)
    return CheckResult(
        name=CHECK_NAME,
        score=round(score, 2),
        label=label_for(score),
        raw_value=round(resolution, 3),
        raw_metric=metric,
        details={
            "document_found": True,
            "doc_type": doc_type,
            "in_frame": not cut_off,
            "corners": [[round(float(x), 1), round(float(y), 1)] for x, y in detection.corners],
            "rotation_deg": round(rotation_degrees(detection), 2),
        },
        retake_message=CUT_OFF_MESSAGE if cut_off else TOO_FAR_MESSAGE,
    )
