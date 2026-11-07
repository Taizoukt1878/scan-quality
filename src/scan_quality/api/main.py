"""FastAPI application exposing the capture-quality gate.

Endpoints:
    POST /score    multipart upload of a JPEG, PNG or PDF -> accept/reject + breakdown
    GET  /health   liveness

This module owns HTTP concerns only: reading the upload, enforcing limits,
mapping domain exceptions onto status codes. All image logic lives in
:mod:`scan_quality.scoring`.
"""

import time
from typing import Literal

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
from scan_quality.scoring.document import DEFAULT_DOCUMENT_TYPE
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

# Document types the caller can declare. Must match the keys of
# scoring.document.DOCUMENT_SIZES_MM; the physical size is what the framing
# check measures resolution against.
DocumentType = Literal["a4", "cheque"]

app = FastAPI(
    title="Scan Quality API",
    version=__version__,
    description=(
        "Decides whether a photo of a document (cheque, claim form, accident "
        "report) is good enough to send to extraction, and if not, what the "
        "user should do differently."
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
    """Report that the service is up."""
    return HealthResponse(status="ok", version=__version__)


@app.post(
    "/score",
    response_model=ScoreResponse,
    summary="Decide whether a document capture is good enough to accept",
    responses={
        400: {"model": ErrorResponse, "description": "Empty or undecodable file"},
        413: {"model": ErrorResponse, "description": "File exceeds the size limit"},
        415: {"model": ErrorResponse, "description": "Not a JPEG, PNG or PDF"},
        503: {"model": ErrorResponse, "description": "poppler unavailable (PDF uploads)"},
    },
)
async def score(
    request: Request,
    file: UploadFile = File(description="Document photo: JPEG, PNG or PDF (first page is scored)."),
    doc_type: DocumentType = Query(
        default=DEFAULT_DOCUMENT_TYPE,
        description="What was photographed; its physical size sets the resolution target.",
    ),
) -> ScoreResponse | JSONResponse:
    """Locate the document, run every check, and return the accept/reject decision.

    File type is determined from the file's own magic bytes, not from its
    extension or declared content type. A PDF page is a render, not a photo,
    so it is scored as the whole document without detection.
    """
    started = time.perf_counter()

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
    # large photo cannot stall every other in-flight request.
    whole_frame = file_meta["source_format"] == "pdf"
    report = await run_in_threadpool(score_image, image, doc_type, whole_frame)

    file_meta = {"filename": file.filename, "size_bytes": len(data), **file_meta}
    total_ms = (time.perf_counter() - started) * 1000.0
    return ScoreResponse.from_report(report, file_meta, total_ms)


@app.exception_handler(404)
async def not_found_handler(request: Request, exc: Exception) -> JSONResponse:
    """Keep 404s in the same error shape as everything else."""
    return _error(404, "not_found", f"No route matches {request.method} {request.url.path}.")
