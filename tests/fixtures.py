"""Synthetic document pages and degradations used by the check tests.

Every test works by taking a clean generated page, applying one controlled
defect, and asserting that the corresponding check notices. This keeps the
suite hermetic — no sample scans committed to the repo — and makes the
relationship between a defect and a score explicit.
"""

import cv2
import numpy as np

PAGE_WIDTH = 1200
PAGE_HEIGHT = 1600
PAPER_VALUE = 245  # near-white, like real scanned paper rather than pure 255
INK_VALUE = 25


def clean_page(
    width: int = PAGE_WIDTH,
    height: int = PAGE_HEIGHT,
    line_count: int = 28,
    text: str = "Devis numero 2024-0187 Total HT 1450,00 EUR",
) -> np.ndarray:
    """Render a sharp, evenly lit, straight page of black text on white.

    Returns a 3-channel BGR array, matching what the image loader produces.
    """
    page = np.full((height, width), PAPER_VALUE, dtype=np.uint8)
    top_margin = int(height * 0.08)
    usable = height - 2 * top_margin
    spacing = usable // max(1, line_count)

    for index in range(line_count):
        baseline_y = top_margin + spacing * (index + 1)
        cv2.putText(
            page,
            text,
            (int(width * 0.06), baseline_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            INK_VALUE,
            2,
            cv2.LINE_AA,
        )
    return cv2.cvtColor(page, cv2.COLOR_GRAY2BGR)


def blurred(image: np.ndarray, kernel: int = 9) -> np.ndarray:
    """Apply a Gaussian blur; larger odd ``kernel`` means more out of focus."""
    return cv2.GaussianBlur(image, (kernel, kernel), 0)


def rotated(image: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate about the centre, filling exposed corners with paper white.

    Positive ``degrees`` rotates counter-clockwise, matching the sign
    convention the skew check reports.
    """
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), degrees, 1.0)
    return cv2.warpAffine(
        image, matrix, (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(PAPER_VALUE, PAPER_VALUE, PAPER_VALUE),
    )


def uneven_lighting(image: np.ndarray, strength: float = 0.55) -> np.ndarray:
    """Multiply in a left-to-right brightness ramp, as from a side-lit photo.

    ``strength`` is the fraction of brightness lost at the darkest edge.
    """
    height, width = image.shape[:2]
    ramp = np.linspace(1.0 - strength, 1.0, width, dtype=np.float32)
    gradient = np.tile(ramp, (height, 1))
    if image.ndim == 3:
        gradient = gradient[:, :, np.newaxis]
    return np.clip(image.astype(np.float32) * gradient, 0, 255).astype(np.uint8)


def low_contrast(image: np.ndarray, factor: float = 0.25) -> np.ndarray:
    """Compress the intensity range toward mid-gray, as in a washed-out scan.

    ``factor`` is the fraction of the original contrast retained.
    """
    mid = 128.0
    return np.clip((image.astype(np.float32) - mid) * factor + mid, 0, 255).astype(np.uint8)


def blank_page(width: int = PAGE_WIDTH, height: int = PAGE_HEIGHT) -> np.ndarray:
    """A featureless white page: no edges, no contrast, no text."""
    return np.full((height, width, 3), PAPER_VALUE, dtype=np.uint8)
