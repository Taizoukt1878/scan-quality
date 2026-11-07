"""Pydantic response models — the public shape of the API.

These exist only to describe and validate what goes over the wire; the scoring
package knows nothing about them.
"""

from typing import Any

from pydantic import BaseModel, Field

from scan_quality.scoring.pipeline import QualityReport


class CheckScore(BaseModel):
    """One capture-quality check."""

    name: str = Field(description="Machine-readable check identifier.")
    score: float = Field(ge=0, le=100, description="Normalized quality, 0-100, higher is better.")
    label: str = Field(description='Human-readable bucket: excellent | good | fair | poor | unknown.')
    raw_value: float | None = Field(
        description="Underlying measurement before normalization; null if unmeasurable."
    )
    raw_metric: str = Field(description="Name and unit of raw_value.")
    details: dict[str, Any] = Field(
        default_factory=dict, description="Check-specific diagnostics. Not used in scoring."
    )
    retake_message: str | None = Field(
        default=None, description="Instruction shown to the user if this check rejects the capture."
    )


class FileMeta(BaseModel):
    """Provenance of the scored page."""

    filename: str | None
    size_bytes: int
    source_format: str = Field(description="jpeg | png | pdf, detected from the file's contents.")
    page_count: int | None = Field(description="Pages in the source document; null if unknown.")
    scored_page: int = Field(description="Which page was scored. Always 1 — PDFs score page 1 only.")
    width: int
    height: int
    render_dpi: int | None = Field(default=None, description="Rasterization DPI, PDFs only.")
    doc_type: str = Field(description="Declared document type: a4 | cheque.")
    document_width: int | None = Field(
        default=None, description="Width of the rectified document in px; null if not found."
    )
    document_height: int | None = Field(
        default=None, description="Height of the rectified document in px; null if not found."
    )


class ScoreResponse(BaseModel):
    """Accept/reject decision and full breakdown for one uploaded capture."""

    accepted: bool = Field(description="True when every check reached its threshold.")
    overall_score: float = Field(ge=0, le=100, description="The lowest check score.")
    overall_label: str
    failed_checks: list[str] = Field(description="Checks below their threshold, worst first.")
    retake_message: str | None = Field(
        description="What the user should do differently; null when accepted."
    )
    checks: list[CheckScore] = Field(
        description="Per-check results. Only framing when no document was found."
    )
    processing_time_ms: float = Field(
        description="Total server-side time: decode, detection and every check."
    )
    file: FileMeta

    @classmethod
    def from_report(
        cls, report: QualityReport, file_meta: dict[str, Any], total_ms: float
    ) -> "ScoreResponse":
        """Build the wire response from a pipeline report plus request context.

        ``total_ms`` is measured by the endpoint and covers file decoding as
        well as scoring, so it is always at least the report's own timing.
        """
        return cls(
            accepted=report.accepted,
            overall_score=report.overall_score,
            overall_label=report.overall_label,
            failed_checks=report.failed_checks,
            retake_message=report.retake_message,
            checks=[CheckScore(**vars(check)) for check in report.checks],
            processing_time_ms=round(total_ms, 2),
            file=FileMeta(**file_meta, **report.meta),
        )


class HealthResponse(BaseModel):
    """Liveness."""

    status: str
    version: str


class ErrorDetail(BaseModel):
    code: str = Field(description="Stable machine-readable error identifier.")
    message: str


class ErrorResponse(BaseModel):
    """Every 4xx and 5xx from this service uses this shape."""

    error: ErrorDetail
