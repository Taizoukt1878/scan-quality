"""Check 5 of 5 — OCR confidence, via Tesseract's per-word certainty.

The other four checks measure proxies for legibility. This one measures the
thing itself: it runs the OCR engine that will actually consume the document
and asks how sure it was. Tesseract reports a 0-100 confidence for every word
it emits — essentially how well the glyph shapes matched its character models
and language dictionary. Averaging those gives a direct, end-to-end read on
whether the scan is machine-readable.

It is the slowest of the five checks by an order of magnitude, and the only
one that depends on an external binary.
"""

from functools import lru_cache

import numpy as np
import pytesseract
from pytesseract import Output, TesseractError, TesseractNotFoundError

from scan_quality.scoring.base import CheckResult, label_for, linear_score, resize_to_max_dim, to_grayscale

# --- Tuning constants -------------------------------------------------------

# Tesseract is trained around 300 DPI; feeding it a 6000px phone photo costs
# seconds of CPU without improving recognition. Cap the longest side.
OCR_MAX_DIM = 2000

# Recognition language. Tesseract needs the matching traineddata pack
# installed. The corpus here (devis/quotes, certificates) is largely French —
# switch to "fra", or "fra+eng" for mixed documents, once that pack is
# installed. Defaults to "eng" because that is the only pack guaranteed to
# ship with every tesseract-ocr distribution.
DEFAULT_LANG = "eng"

# --psm 3 = fully automatic page segmentation without orientation detection.
# The right mode for a whole scanned page; the single-block and single-line
# modes would mis-segment multi-column quotes.
TESSERACT_CONFIG = "--psm 3"

# Tokens shorter than this are dropped before averaging. Speckle and JPEG
# artefacts routinely register as one-character "words" with low confidence;
# including them would make a noisy-but-readable scan look worse than it is.
MIN_WORD_LENGTH = 2

# Mean word confidence that scores 0. Tesseract's confidence does not use the
# bottom of its own range for real text — genuine words on a poor scan bottom
# out around here, and anything lower usually means it is reading noise.
OCR_CONF_FLOOR = 40.0

# Mean word confidence that scores 100. A clean scan of printed text averages
# in the high 80s / low 90s; demanding a perfect 100 would be unreachable
# because proper nouns and reference numbers are always out-of-dictionary and
# score lower no matter how good the image is.
OCR_CONF_CEILING = 92.0

# Score when Tesseract recognizes no usable words at all. Unlike the skew
# check's neutral fallback, this is a genuine finding rather than a
# measurement failure: whatever the cause — blank page, wrong language, an
# image far too degraded to segment — the document is not machine-readable.
NO_TEXT_SCORE = 0.0

CHECK_NAME = "ocr_confidence"


class OcrUnavailableError(RuntimeError):
    """The Tesseract binary is missing, or failed for an unexpected reason.

    A deployment problem, not a property of the uploaded image, so it is
    raised rather than folded into a low score.
    """


class OcrLanguageError(ValueError):
    """A requested language pack is not installed.

    Distinct from :class:`OcrUnavailableError` because it is caused by the
    caller's request rather than by a broken environment, and is fixed by
    installing the pack (or asking for a different one).
    """


def is_tesseract_available() -> bool:
    """Return True if the Tesseract binary can be invoked."""
    try:
        pytesseract.get_tesseract_version()
    except (TesseractNotFoundError, OSError):
        return False
    return True


@lru_cache(maxsize=1)
def available_languages() -> frozenset[str]:
    """Return the language packs Tesseract has installed.

    Cached because it costs a subprocess call and the installed packs cannot
    change while the process is running.

    Raises:
        OcrUnavailableError: if the Tesseract binary cannot be found.
    """
    try:
        return frozenset(pytesseract.get_languages(config=""))
    except TesseractNotFoundError as exc:
        raise OcrUnavailableError(
            "Tesseract OCR binary not found on PATH. See README.md for install commands."
        ) from exc


def _require_languages(lang: str) -> None:
    """Check every pack in a ``"fra+eng"``-style spec is installed.

    Validated up front rather than left to Tesseract, which otherwise fails
    mid-run with a message about TESSDATA_PREFIX that tells the caller nothing
    actionable.

    Raises:
        OcrLanguageError: if any requested pack is missing.
    """
    installed = available_languages()
    missing = sorted(set(lang.split("+")) - installed)
    if missing:
        raise OcrLanguageError(
            f"Tesseract language pack(s) not installed: {', '.join(missing)}. "
            f"Available: {', '.join(sorted(installed))}. "
            "See README.md for how to install additional packs."
        )


def mean_word_confidence(image: np.ndarray, lang: str = DEFAULT_LANG) -> tuple[float | None, int]:
    """Run OCR and return ``(mean_confidence, word_count)``.

    Returns ``(None, 0)`` when no token survives filtering. Tokens are kept
    only when Tesseract reports a non-negative confidence (it uses -1 for the
    layout blocks it emits alongside words) and the stripped text is at least
    :data:`MIN_WORD_LENGTH` characters.

    Raises:
        OcrUnavailableError: if the Tesseract binary cannot be found, or the
            OCR run fails for an unexpected reason.
        OcrLanguageError: if ``lang`` names a pack that is not installed.
    """
    _require_languages(lang)
    gray = resize_to_max_dim(to_grayscale(image), OCR_MAX_DIM)
    try:
        data = pytesseract.image_to_data(
            gray, lang=lang, config=TESSERACT_CONFIG, output_type=Output.DICT
        )
    except TesseractNotFoundError as exc:
        raise OcrUnavailableError(
            "Tesseract OCR binary not found on PATH. See README.md for install commands."
        ) from exc
    except TesseractError as exc:
        # Anything else the engine rejects — a corrupt traineddata file, an
        # unsupported config. Wrapped so it surfaces as a service error rather
        # than escaping the API as an unhandled 500.
        raise OcrUnavailableError(f"Tesseract failed to process the image: {exc}") from exc

    confidences = [
        float(conf)
        for conf, text in zip(data["conf"], data["text"])
        if float(conf) >= 0.0 and len(str(text).strip()) >= MIN_WORD_LENGTH
    ]
    if not confidences:
        return None, 0
    return float(np.mean(confidences)), len(confidences)


def score_ocr_confidence(image: np.ndarray, lang: str = DEFAULT_LANG) -> CheckResult:
    """Score machine-readability from 0 (unreadable) to 100 (clean text).

    The mean per-word confidence is remapped linearly from
    :data:`OCR_CONF_FLOOR` (0) to :data:`OCR_CONF_CEILING` (100) rather than
    used as the score directly. Tesseract's raw confidence only occupies the
    top half of its nominal range for real documents, so passing it straight
    through would compress every result into 40-95 and make the check far less
    discriminating than the other four.

    Raises:
        OcrUnavailableError: if the Tesseract binary cannot be found or fails.
        OcrLanguageError: if ``lang`` names a pack that is not installed.
    """
    confidence, word_count = mean_word_confidence(image, lang=lang)

    if confidence is None:
        return CheckResult(
            name=CHECK_NAME,
            score=NO_TEXT_SCORE,
            label=label_for(NO_TEXT_SCORE),
            raw_value=None,
            raw_metric="mean per-word Tesseract confidence (0-100)",
            details={
                "word_count": 0,
                "lang": lang,
                "reason": "Tesseract recognized no words of usable length",
            },
        )

    score = linear_score(confidence, OCR_CONF_FLOOR, OCR_CONF_CEILING)
    return CheckResult(
        name=CHECK_NAME,
        score=round(score, 2),
        label=label_for(score),
        raw_value=round(confidence, 2),
        raw_metric="mean per-word Tesseract confidence (0-100)",
        details={"word_count": word_count, "lang": lang},
    )
