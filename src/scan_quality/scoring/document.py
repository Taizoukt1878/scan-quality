"""Step 0 — find the document in the photo and flatten it to a known size.

A phone capture is mostly *not* the document: it is the table, the user's
hand, the car seat. Scoring the whole frame would let the table's wood grain
count as sharpness and would judge a cheque that fills a third of the frame
at the wrong resolution. So before any check runs, the page is located,
its four corners are found, and a perspective warp maps it onto a rectangle
of the document's real physical size at :data:`PX_PER_MM`.

Every pixel check then measures the same thing on every capture: a flat,
straight document at a fixed resolution. That is also why rotation and
perspective are not scored — this step removes them.

Detection assumes the usual capture setup: a light document on a darker,
plainer surface. A white page on a white desk, or a dark card on a light one,
will not be found, and the capture is rejected with an instruction to fix
exactly that.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from scan_quality.scoring.base import resize_to_max_dim, to_grayscale

# --- Tuning constants -------------------------------------------------------

# Working resolution of the rectified document: ~200 DPI. Enough for printed
# body text and handwriting, and the target that the framing check measures
# captures against.
PX_PER_MM = 8.0

# Physical size of each supported document type, (short side, long side) in
# mm. Orientation is taken from the photo, so landscape and portrait captures
# of the same type both work.
DOCUMENT_SIZES_MM: dict[str, tuple[float, float]] = {
    "a4": (210.0, 297.0),
    "cheque": (80.0, 175.0),  # French cheque format
}
DEFAULT_DOCUMENT_TYPE = "a4"

# Contour finding runs on a downscaled copy: corner positions only need to be
# accurate to a few pixels, and the cost of thresholding a 12 MP frame buys
# nothing.
DETECTION_MAX_DIM = 1000

# The paper is separated from the surface by a threshold placed this far from
# the surface's brightness toward the paper's. Low on purpose: a page half in
# deep shadow is still far brighter than the table it lies on, and a
# threshold half-way up (as Otsu tends to place it) would cut the shadowed
# half off.
SURFACE_GAP_FRAC = 0.2

# Minimum brightness gap between surface and paper. Below it the page cannot
# be told from what it lies on: a white page on a white desk.
MIN_SURFACE_GAP = 30.0

# Width of the frame border sampled to measure the surface, as a fraction of
# the frame, and the percentile of that band taken as the surface level. A
# low percentile so that a page overflowing one or two sides of the frame —
# its paper filling part of the band — still leaves the surface measurable on
# the others, and is then reported as cut off rather than as not found.
SURFACE_BAND_FRAC = 0.02
SURFACE_PERCENTILE = 25.0

# Closing applied to the paper mask before contour finding, so lines of text
# near the page edge do not bite notches into its outline. In detection-size
# pixels.
DETECTION_CLOSE_PX = 15

# The document must cover at least this share of the frame. Anything smaller
# is either not the document (a sticker, a reflection) or so far away that the
# framing check would reject it anyway.
MIN_DOCUMENT_AREA_FRAC = 0.15

# Polygon simplification tolerance, as a fraction of the outline's perimeter.
# Large enough to straighten a slightly curled edge into one side, small
# enough that a folded corner still shows up as a fifth vertex.
APPROX_EPSILON_FRAC = 0.02

# Band trimmed off every edge of the rectified document. Detected corners are
# accurate to a few pixels, so the outermost rows still blend paper with the
# surface around it — a thin dark line the contrast check would read as faint
# ink. 8 px is 1 mm; real content does not sit that close to the edge.
EDGE_TRIM_PX = 8


@dataclass
class DocumentDetection:
    """Where the document sits in the original photo.

    Attributes:
        corners: (4, 2) float32 array of pixel positions, ordered top-left,
            top-right, bottom-right, bottom-left.
        frame_size: ``(width, height)`` of the original photo.
        whole_frame: True when the frame *is* the document (a rendered PDF
            page) rather than a photo of one. Edge-margin rules do not apply.
    """

    corners: np.ndarray
    frame_size: tuple[int, int]
    whole_frame: bool = False

    def side_lengths_px(self) -> np.ndarray:
        """Return the lengths of the top, right, bottom and left sides."""
        return np.linalg.norm(self.corners - np.roll(self.corners, -1, axis=0), axis=1)

    def is_landscape(self) -> bool:
        """True when the document appears wider than it is tall in the photo."""
        top, right, bottom, left = self.side_lengths_px()
        return (top + bottom) >= (left + right)


def order_corners(points: np.ndarray) -> np.ndarray:
    """Order four points as top-left, top-right, bottom-right, bottom-left.

    The top-left corner has the smallest x + y and the bottom-right the
    largest; top-right has the smallest y - x and bottom-left the largest.
    Holds for any tilt below 45 degrees, which covers every hand-held capture.
    """
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    sums = points.sum(axis=1)
    diffs = points[:, 1] - points[:, 0]
    return np.array(
        [
            points[np.argmin(sums)],
            points[np.argmin(diffs)],
            points[np.argmax(sums)],
            points[np.argmax(diffs)],
        ],
        dtype=np.float32,
    )


def whole_frame_detection(image: np.ndarray) -> DocumentDetection:
    """Treat the entire image as the document (for rendered PDF pages)."""
    height, width = image.shape[:2]
    corners = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32
    )
    return DocumentDetection(corners=corners, frame_size=(width, height), whole_frame=True)


def detect_document(image: np.ndarray) -> DocumentDetection | None:
    """Find the document's four corners in a photo, or ``None``.

    The surface's brightness is read from the frame border and the paper's
    from the brightest pixels; everything a little brighter than the surface
    (see :data:`SURFACE_GAP_FRAC`) counts as paper. The largest paper region
    is taken as the document and its convex outline is simplified to a
    polygon. Only a clean quadrilateral counts: a region with more or fewer
    corners is a partial page, a folded one, or not a page at all.
    """
    gray = to_grayscale(image)
    height, width = gray.shape[:2]
    small = resize_to_max_dim(gray, DETECTION_MAX_DIM)
    scale_x = width / small.shape[1]
    scale_y = height / small.shape[0]

    blurred = cv2.GaussianBlur(small, (5, 5), 0)
    band_h = max(1, int(blurred.shape[0] * SURFACE_BAND_FRAC))
    band_w = max(1, int(blurred.shape[1] * SURFACE_BAND_FRAC))
    border = np.concatenate([
        blurred[:band_h].ravel(), blurred[-band_h:].ravel(),
        blurred[:, :band_w].ravel(), blurred[:, -band_w:].ravel(),
    ])
    surface = float(np.percentile(border, SURFACE_PERCENTILE))
    paper = float(np.percentile(blurred, 99))
    if paper - surface < MIN_SURFACE_GAP:
        return None
    threshold = surface + SURFACE_GAP_FRAC * (paper - surface)
    mask = (blurred > threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (DETECTION_CLOSE_PX, DETECTION_CLOSE_PX))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    outline = max(contours, key=cv2.contourArea)
    if cv2.contourArea(outline) < MIN_DOCUMENT_AREA_FRAC * small.shape[0] * small.shape[1]:
        return None

    hull = cv2.convexHull(outline)
    polygon = cv2.approxPolyDP(hull, APPROX_EPSILON_FRAC * cv2.arcLength(hull, True), True)
    if len(polygon) != 4:
        return None

    corners = polygon.reshape(4, 2).astype(np.float32) * np.array([scale_x, scale_y], dtype=np.float32)
    return DocumentDetection(corners=order_corners(corners), frame_size=(width, height))


def physical_size_mm(detection: DocumentDetection, doc_type: str) -> tuple[float, float]:
    """Return the document's ``(width_mm, height_mm)`` as it lies in the photo."""
    short_mm, long_mm = DOCUMENT_SIZES_MM[doc_type]
    return (long_mm, short_mm) if detection.is_landscape() else (short_mm, long_mm)


def side_resolutions_px_per_mm(detection: DocumentDetection, doc_type: str) -> np.ndarray:
    """Return px/mm along the top, right, bottom and left sides of the document."""
    width_mm, height_mm = physical_size_mm(detection, doc_type)
    return detection.side_lengths_px() / np.array([width_mm, height_mm, width_mm, height_mm])


def rectify(image: np.ndarray, detection: DocumentDetection, doc_type: str) -> np.ndarray:
    """Warp the detected document onto a flat rectangle at :data:`PX_PER_MM`.

    When the photo holds more detail than the working resolution, it is first
    shrunk with area averaging. A perspective warp only point-samples, and
    point-sampling a much finer image aliases — it would manufacture edges
    that inflate the sharpness score.

    The result is trimmed by :data:`EDGE_TRIM_PX` on every side.
    """
    width_mm, height_mm = physical_size_mm(detection, doc_type)
    out_width = int(round(width_mm * PX_PER_MM))
    out_height = int(round(height_mm * PX_PER_MM))

    corners = detection.corners.copy()
    shrink = PX_PER_MM / float(side_resolutions_px_per_mm(detection, doc_type).mean())
    if shrink < 1.0:
        image = cv2.resize(image, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_AREA)
        corners *= shrink

    target = np.array(
        [[0, 0], [out_width - 1, 0], [out_width - 1, out_height - 1], [0, out_height - 1]],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(corners, target)
    flat = cv2.warpPerspective(image, homography, (out_width, out_height), flags=cv2.INTER_LINEAR)
    return flat[EDGE_TRIM_PX:out_height - EDGE_TRIM_PX, EDGE_TRIM_PX:out_width - EDGE_TRIM_PX]
