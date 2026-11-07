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


def sensor_noise(image: np.ndarray, sigma: float = 3.0, seed: int = 0) -> np.ndarray:
    """Add Gaussian grain, as every real camera sensor does.

    Noise matters to the contrast check, which measures ink against the
    paper's own noise: a noiseless synthetic page would pass at any contrast.
    """
    rng = np.random.default_rng(seed)
    noisy = image.astype(np.float32) + rng.normal(0.0, sigma, image.shape).astype(np.float32)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def hard_shadow(image: np.ndarray, start_frac: float = 0.5, darkness: float = 0.5) -> np.ndarray:
    """Darken everything below ``start_frac`` of the height, like an arm's shadow.

    ``darkness`` is the fraction of light the shadowed part still receives.
    """
    shadowed = image.astype(np.float32)
    start = int(image.shape[0] * start_frac)
    shadowed[start:] *= darkness
    return np.clip(shadowed, 0, 255).astype(np.uint8)


def glare_spot(image: np.ndarray, center_frac=(0.6, 0.4), size_px=(100, 60)) -> np.ndarray:
    """Paint a saturated white ellipse, like a ceiling light reflected off gloss.

    ``center_frac`` is ``(x, y)`` as fractions of the image; ``size_px`` is the
    ellipse's full ``(width, height)``.
    """
    spotted = image.copy()
    height, width = image.shape[:2]
    center = (int(width * center_frac[0]), int(height * center_frac[1]))
    axes = (size_px[0] // 2, size_px[1] // 2)
    colour = (255, 255, 255) if image.ndim == 3 else 255
    cv2.ellipse(spotted, center, axes, 0, 0, 360, colour, -1)
    return spotted


def photo_of(
    page: np.ndarray,
    margin_frac: float = 0.1,
    surface_value: int = 70,
    tilt_px: int = 0,
    noise_sigma: float = 3.0,
) -> np.ndarray:
    """Simulate a phone photo: the page lying on a darker surface.

    ``margin_frac`` is the surface visible around the page, as a fraction of
    the page size; a negative value crops the page. ``tilt_px`` pulls the top
    corners inward, as when the phone is not held parallel to the page.
    """
    height, width = page.shape[:2]
    margin_x, margin_y = int(width * margin_frac), int(height * margin_frac)
    frame_w, frame_h = width + 2 * margin_x, height + 2 * margin_y
    source = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    target = source + np.array([margin_x, margin_y], dtype=np.float32)
    target[0, 0] += tilt_px
    target[1, 0] -= tilt_px
    matrix = cv2.getPerspectiveTransform(source, target)
    photo = cv2.warpPerspective(
        page, matrix, (frame_w, frame_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(surface_value,) * 3,
    )
    return sensor_noise(photo, noise_sigma) if noise_sigma > 0 else photo


def a4_page() -> np.ndarray:
    """A clean A4 page at the 8 px/mm working resolution (1680 x 2376)."""
    return clean_page(width=1680, height=2376, line_count=40)
