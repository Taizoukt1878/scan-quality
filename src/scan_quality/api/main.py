"""FastAPI application exposing the scan-quality scorer.

Endpoints:
    POST /score    multipart upload of a JPEG, PNG or PDF -> scoring breakdown
    GET  /health   liveness plus external-binary status

This module owns HTTP concerns only: reading the upload, enforcing limits,
mapping domain exceptions onto status codes. All image logic lives in
:mod:`scan_quality.scoring`.
"""

import re
import time

from fastapi import FastAPI, File, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from scan_quality import __version__
from scan_quality.api.schemas import ErrorResponse, HealthResponse, ScoreResponse
from scan_quality.loaders import (
    CorruptFileError,
    PdfRenderError,
    UnsupportedFileTypeError,
    load_page_image,
)
from scan_quality.scoring.ocr import (
    DEFAULT_LANG,
    OcrLanguageError,
    OcrUnavailableError,
    is_tesseract_available,
)
from scan_quality.scoring.pipeline import score_image

# --- Request limits ---------------------------------------------------------

# Largest upload accepted. A 300-DPI colour A4 scan is roughly 25 MB
# uncompressed but only 3-6 MB as JPEG, and a multi-page PDF of such scans
# rarely exceeds this; the cap exists to stop a single request exhausting
# memory, since the file is buffered before decoding.
MAX_UPLOAD_BYTES = 20 * 1024 * 1024

# Upload is consumed in chunks so the size limit can be enforced before the
# whole body is in memory, rather than after.
UPLOAD_CHUNK_BYTES = 1024 * 1024

# Tesseract language codes: three lowercase letters, optionally combined with
# "+" (e.g. "fra+eng"). Validated because this value reaches a subprocess
# argument list.
LANG_PATTERN = re.compile(r"^[a-z]{3}(\+[a-z]{3})*$")

app = FastAPI(
    title="Scan Quality API",
    version=__version__,
    description=(
        "Scores the scan quality of a document image (cheques, certificates, "
        "devis/quotes) using five independent, locally-run checks."
    ),
)

# CORS is wide open so the service can be called from any local front-end
# during development.
# TODO: Before deploying anywhere shared, restrict allow_origins to the exact
# front-end origins that should be able to call this service, and narrow
# allow_methods to ["GET", "POST"].
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _error(status_code: int, code: str, message: str) -> JSONResponse:
    """Build the service's uniform error body."""
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse.model_validate({"error": {"code": code, "message": message}}).model_dump(),
    )


class _UploadTooLargeError(ValueError):
    """Internal signal that the upload exceeded :data:`MAX_UPLOAD_BYTES`."""


async def _read_upload(upload: UploadFile) -> bytes:
    """Read the upload, aborting as soon as it exceeds the size limit."""
    chunks: list[bytes] = []
    total = 0
    while chunk := await upload.read(UPLOAD_CHUNK_BYTES):
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise _UploadTooLargeError
        chunks.append(chunk)
    return b"".join(chunks)


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness check",
)
def health() -> HealthResponse:
    """Report that the service is up, and whether Tesseract can be reached.

    ``tesseract_available: false`` means ``/score`` will return 503 for every
    request, so this is worth alerting on.
    """
    available = is_tesseract_available()
    version: str | None = None
    if available:
        import pytesseract

        version = str(pytesseract.get_tesseract_version())
    return HealthResponse(
        status="ok",
        version=__version__,
        tesseract_available=available,
        tesseract_version=version,
    )


@app.post(
    "/score",
    response_model=ScoreResponse,
    summary="Score the scan quality of an uploaded document",
    responses={
        400: {
            "model": ErrorResponse,
            "description": "Empty/undecodable file, or a bad or uninstalled language",
        },
        413: {"model": ErrorResponse, "description": "File exceeds the size limit"},
        415: {"model": ErrorResponse, "description": "Not a JPEG, PNG or PDF"},
        503: {"model": ErrorResponse, "description": "Tesseract or poppler unavailable"},
    },
)
async def score(
    request: Request,
    file: UploadFile = File(description="Document image: JPEG, PNG or PDF (first page is scored)."),
    lang: str = Query(
        default=DEFAULT_LANG,
        description='Tesseract language pack(s), e.g. "fra" or "fra+eng". Must be installed.',
    ),
) -> ScoreResponse | JSONResponse:
    """Run the five quality checks over one uploaded page.

    File type is determined from the file's own magic bytes, not from its
    extension or declared content type.
    """
    started = time.perf_counter()

    if not LANG_PATTERN.match(lang):
        return _error(
            400,
            "invalid_language",
            f'Invalid language code "{lang}". Expected three lowercase letters, '
            'optionally joined with "+", e.g. "fra" or "fra+eng".',
        )

    try:
        data = await _read_upload(file)
    except _UploadTooLargeError:
        limit_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
        return _error(413, "file_too_large", f"File exceeds the {limit_mb} MB limit.")

    try:
        image, file_meta = await run_in_threadpool(load_page_image, data)
    except UnsupportedFileTypeError as exc:
        return _error(415, "unsupported_file_type", str(exc))
    except CorruptFileError as exc:
        return _error(400, "corrupt_file", str(exc))
    except PdfRenderError as exc:
        return _error(503, "pdf_render_unavailable", str(exc))

    # Scoring is CPU-bound and synchronous; off the event loop it goes, so one
    # slow OCR pass cannot stall every other in-flight request.
    try:
        report = await run_in_threadpool(score_image, image, lang)
    except OcrLanguageError as exc:
        # The caller asked for a pack this server does not have. Their problem
        # to fix (or the operator's), so 400 rather than 503.
        return _error(400, "language_not_available", str(exc))
    except OcrUnavailableError as exc:
        return _error(503, "ocr_unavailable", str(exc))

    file_meta = {"filename": file.filename, "size_bytes": len(data), **file_meta}
    total_ms = (time.perf_counter() - started) * 1000.0
    return ScoreResponse.from_report(report, file_meta, total_ms)


@app.exception_handler(404)
async def not_found_handler(request: Request, exc: Exception) -> JSONResponse:
    """Keep 404s in the same error shape as everything else."""
    return _error(404, "not_found", f"No route matches {request.method} {request.url.path}.")
