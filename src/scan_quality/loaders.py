"""Turn an uploaded file's bytes into a decoded page image.

Accepts JPEG, PNG and PDF. For a PDF only the first page is rendered — the
service scores one page per request, and for the document types in scope
(cheques, certificates, quotes) the first page is the one that carries the
information worth checking.

File type is decided by magic bytes rather than by the filename extension or
the client-supplied content type, both of which are trivially wrong and
trivially forged.
"""

from typing import Any

import cv2
import numpy as np

# --- Constants --------------------------------------------------------------

# Leading bytes that identify each accepted format.
PDF_MAGIC = b"%PDF"
JPEG_MAGIC = b"\xff\xd8\xff"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

# Resolution at which PDF pages are rasterized. 200 DPI is ~7.9 px/mm, the
# working resolution every check is calibrated at (scoring.document.PX_PER_MM),
# so a rendered page is scored at the same detail as a good phone capture
# without paying for 300 DPI in memory and render time.
PDF_RENDER_DPI = 200


class UnsupportedFileTypeError(ValueError):
    """The uploaded bytes are not a JPEG, PNG or PDF."""


class CorruptFileError(ValueError):
    """The file has a recognized type but could not be decoded."""


class PdfRenderError(RuntimeError):
    """Poppler is missing or failed while rasterizing the PDF.

    An environment fault rather than a problem with the upload, so it is
    distinct from :class:`CorruptFileError`.
    """


def detect_format(data: bytes) -> str:
    """Return ``"pdf"``, ``"jpeg"`` or ``"png"`` based on the leading bytes.

    Raises:
        UnsupportedFileTypeError: if the signature matches none of them.
    """
    if data.startswith(PDF_MAGIC):
        return "pdf"
    if data.startswith(JPEG_MAGIC):
        return "jpeg"
    if data.startswith(PNG_MAGIC):
        return "png"
    raise UnsupportedFileTypeError(
        "File is not a JPEG, PNG or PDF (its contents do not match any of those formats)."
    )


def _decode_raster(data: bytes) -> np.ndarray:
    """Decode JPEG/PNG bytes to a 3-channel BGR array."""
    buffer = np.frombuffer(data, dtype=np.uint8)
    # IMREAD_COLOR drops any alpha channel and gives every downstream check a
    # predictable 3-channel BGR array regardless of how the source was saved.
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise CorruptFileError("Image file could not be decoded; it may be truncated or corrupt.")
    return image


def _render_pdf_first_page(data: bytes) -> tuple[np.ndarray, int | None]:
    """Rasterize page 1 of a PDF; return ``(bgr_image, total_page_count)``.

    ``total_page_count`` is ``None`` when poppler could read the page but not
    the document metadata.
    """
    # Imported lazily so that a machine without poppler can still use the
    # service for JPEG and PNG uploads.
    from pdf2image import convert_from_bytes, pdfinfo_from_bytes
    from pdf2image.exceptions import PDFInfoNotInstalledError, PDFPageCountError, PDFSyntaxError

    try:
        pages = convert_from_bytes(
            data, dpi=PDF_RENDER_DPI, first_page=1, last_page=1, fmt="ppm"
        )
    except PDFInfoNotInstalledError as exc:
        raise PdfRenderError(
            "poppler is not installed, so PDF files cannot be rendered. "
            "See README.md for install commands."
        ) from exc
    except (PDFPageCountError, PDFSyntaxError) as exc:
        raise CorruptFileError(f"PDF could not be rendered: {exc}") from exc

    if not pages:
        raise CorruptFileError("PDF contains no pages.")

    page_count: int | None = None
    try:
        page_count = int(pdfinfo_from_bytes(data)["Pages"])
    except Exception:  # noqa: BLE001 - metadata is a nicety, never fatal
        page_count = None

    # PIL gives RGB; every check downstream assumes OpenCV's BGR ordering.
    rgb = np.asarray(pages[0].convert("RGB"), dtype=np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), page_count


def load_page_image(data: bytes) -> tuple[np.ndarray, dict[str, Any]]:
    """Decode uploaded bytes into a page image plus provenance metadata.

    Args:
        data: Raw bytes of the uploaded file.

    Returns:
        ``(image, meta)`` where ``image`` is a 3-channel BGR uint8 array and
        ``meta`` records the detected ``source_format``, the ``page_count``
        for PDFs (``None`` if unknown, ``1`` for raster uploads) and the
        ``scored_page`` — always 1, stated explicitly so a caller can see that
        later PDF pages were not examined.

    Raises:
        UnsupportedFileTypeError: unrecognized format.
        CorruptFileError: recognized format that would not decode.
        PdfRenderError: poppler missing or broken.
    """
    if not data:
        raise CorruptFileError("Uploaded file is empty.")

    file_format = detect_format(data)
    if file_format == "pdf":
        image, page_count = _render_pdf_first_page(data)
        meta: dict[str, Any] = {
            "source_format": "pdf",
            "page_count": page_count,
            "scored_page": 1,
            "render_dpi": PDF_RENDER_DPI,
        }
    else:
        image = _decode_raster(data)
        meta = {"source_format": file_format, "page_count": 1, "scored_page": 1}

    return image, meta
