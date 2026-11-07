"""Pure capture-quality checks. No FastAPI, no I/O — arrays in, scores out."""

from scan_quality.scoring.base import CheckResult, label_for

__all__ = ["CheckResult", "label_for"]
