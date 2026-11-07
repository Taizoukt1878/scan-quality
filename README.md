# Scan Quality API

Decides whether a **photo of a document** — a cheque, an accident report, a claim form, an invoice —
is good enough to send to extraction, and if not, tells the user what to do differently. Built for
capture flows in insurance and banking, where a rejected photo is retaken on the spot instead of
failing silently downstream.

Everything runs locally on free, open-source software. No external APIs, no network calls, no keys,
no OCR engine.

Upload a JPEG, PNG or PDF; get back an accept/reject decision, the instruction to show the user,
and a per-check breakdown with the raw measurements behind each score.

The formulas, their reasoning and worked examples are in **[docs/scores.md](docs/scores.md)**.

---

## How it works

1. **Find the document.** The page is located against the surface it lies on, its four corners
   are found, and it is warped flat onto its real physical size at 8 px/mm (~200 DPI). Rotation and
   perspective are removed here, so they are never penalized. PDF pages skip this step.
2. **Run five checks** on the flattened document, each normalized to 0–100 where **higher is
   better**:

| Check | Catches | Raw metric | Scores 0 at | Scores 100 at | Module |
|-------|---------|------------|-------------|---------------|--------|
| **Framing** | Missing, cut-off or too-distant document | Worst-side resolution, px/mm | `4.0` (or cut off) | `8.0` | `scoring/framing.py` |
| **Sharpness** | Hand shake, focus falloff | 10th-percentile tile ratio `Var(∇²)/Var(I)` | `0.005` | `0.10` (log ramp) | `scoring/sharpness.py` |
| **Shadows** | Shadow of a hand or phone | Paper brightness ratio p5/p95 | `0.40` | `0.80` | `scoring/shadows.py` |
| **Glare** | Reflections, overexposure | Largest saturated blob, % of area | `1.0 %` | `0.05 %` | `scoring/glare.py` |
| **Contrast** | Faint ink, low-light noise | Worst-cell ink/paper gap in noise units (+ ink depth) | `2.0σ` | `6.0σ` | `scoring/contrast.py` |

3. **Gate on the weakest link.** The capture is accepted only if **every** check reaches its
   threshold (50 by default). The overall score is the **minimum** check score, and the worst
   failing check supplies the retake message. A mean would let four good scores outvote one fatal
   defect.

### Labels

The same vocabulary applies to every check and to the overall score:

| Score | Label |
|-------|-------|
| ≥ 85 | `excellent` |
| ≥ 70 | `good` |
| ≥ 50 | `fair` |
| < 50 | `poor` → rejected |

Sharpness reports `unknown` (a neutral 50) on a page with no text to judge focus on; the contrast
check rejects such a page instead, with the right instruction.

---

## Prerequisites

One **system** package is needed, and only for PDF uploads. `pdf2image` is a thin wrapper that
shells out to it.

| Binary | Needed for | Without it |
|--------|-----------|------------|
| `poppler` (`pdftoppm`, `pdfinfo`) | Rendering PDF uploads | PDF uploads return **503**; JPEG/PNG still work |

```bash
sudo apt-get install -y poppler-utils   # Ubuntu / Debian
brew install poppler                    # macOS
```

---

## Install and run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

uvicorn scan_quality.api.main:app --reload --port 8000
```

Interactive API docs: <http://127.0.0.1:8000/docs>

### Docker

```bash
docker build -t scan-quality .
docker run --rm -p 8000:8000 scan-quality
```

---

## API

### `POST /score`

`multipart/form-data` upload of a JPEG, PNG or PDF.

| Field | In | Required | Description |
|-------|-----|----------|-------------|
| `file` | form-data | yes | The capture. For a PDF, **only page 1 is scored**, as the whole document. |
| `doc_type` | query | no | `a4` (default) or `cheque`. Its physical size sets the resolution target and the flattened size. |

```bash
curl -X POST "http://127.0.0.1:8000/score?doc_type=a4" -F "file=@claim.jpg"
```

The file type is decided by the file's **magic bytes**, not its extension or declared content type.

**Example response** — a claim form with a ceiling-light reflection (detail lists shortened):

```json
{
  "accepted": false,
  "overall_score": 6.29,
  "overall_label": "poor",
  "failed_checks": ["glare"],
  "retake_message": "Tilt the phone slightly to remove the reflection.",
  "checks": [
    {
      "name": "framing",
      "score": 99.84,
      "label": "excellent",
      "raw_value": 7.993,
      "raw_metric": "worst-side resolution (px per mm of document)",
      "details": { "document_found": true, "doc_type": "a4", "in_frame": true, "rotation_deg": 0.0, "duration_ms": 13.59 },
      "retake_message": "Move closer to the document."
    },
    {
      "name": "sharpness",
      "score": 95.07,
      "label": "excellent",
      "raw_value": 0.0863,
      "raw_metric": "10th-percentile tile ratio Var(Laplacian)/Var(intensity)",
      "details": { "text_tiles": 341, "tile_px": 64, "median_ratio": 0.1103, "duration_ms": 87.48 },
      "retake_message": "Hold the phone still and parallel to the document."
    },
    {
      "name": "shadows",
      "score": 100.0,
      "label": "excellent",
      "raw_value": 0.996,
      "raw_metric": "paper brightness ratio p5/p95 (0-1)",
      "details": { "dark_paper": 246.0, "bright_paper": 247.0, "duration_ms": 100.57 },
      "retake_message": "Move to even light and avoid casting a shadow on the document."
    },
    {
      "name": "glare",
      "score": 6.29,
      "label": "poor",
      "raw_value": 0.9402,
      "raw_metric": "largest saturated blob (% of document area)",
      "details": { "largest_blob_px": 36922, "blob_count": 1, "duration_ms": 55.47 },
      "retake_message": "Tilt the phone slightly to remove the reflection."
    },
    {
      "name": "contrast",
      "score": 100.0,
      "label": "excellent",
      "raw_value": 91.5,
      "raw_metric": "worst-cell ink/paper gap in paper-noise units",
      "details": { "worst_ink_depth": 0.7262, "cells_measured": 8, "grid": "4x4", "duration_ms": 213.72 },
      "retake_message": "Find more light; the text is too faint to read."
    }
  ],
  "processing_time_ms": 512.24,
  "file": {
    "filename": "claim.jpg",
    "size_bytes": 1055284,
    "source_format": "jpeg",
    "page_count": 1,
    "scored_page": 1,
    "width": 2016,
    "height": 2850,
    "render_dpi": null,
    "doc_type": "a4",
    "document_width": 1664,
    "document_height": 2360
  }
}
```

The other four checks average above 95, yet the capture is rejected: the reflection covers ~0.9 %
of the page, enough to hide a field.

Every check carries a `retake_message`; only the worst failing check's is promoted to the top level.
When no document is found, `checks` holds only `framing`.

`raw_value` is the measurement *before* normalization. It is in the response on purpose: it is what
you need to re-tune the thresholds against your own captures without re-instrumenting the service.

**Retry budget.** The API is stateless. The capturing app should count attempts and, after three
rejections, accept the best capture so far and flag it for manual review, so a user filing a claim
never gets stuck in a loop.

### `GET /health`

```json
{ "status": "ok", "version": "0.2.0" }
```

### Errors

All errors share one shape:

```json
{ "error": { "code": "unsupported_file_type", "message": "File is not a JPEG, PNG or PDF ..." } }
```

| Status | `code` | Cause |
|--------|--------|-------|
| 400 | `corrupt_file` | Empty, truncated or undecodable file |
| 413 | `file_too_large` | Upload exceeds 20 MB |
| 415 | `unsupported_file_type` | Contents are not JPEG, PNG or PDF |
| 422 | — | Missing `file`, or `doc_type` not `a4`/`cheque` (FastAPI's own validation) |
| 503 | `pdf_render_unavailable` | poppler not found |

A rejected capture is **not** an error: it is a 200 with `"accepted": false`.

---

## Tuning

Every threshold is a named constant at the top of its check's module, with a comment explaining
the choice; the per-check accept thresholds are `ACCEPT_THRESHOLDS` in `scoring/pipeline.py`. To
recalibrate, collect real captures labelled by their downstream outcome, read the `raw_value`
fields off the responses, and move the anchors.

The anchors were set against **synthetic** test captures, not real phone photos. See
[docs/scores.md → Calibration](docs/scores.md#calibration) for the procedure and the risks to check
first (auto-exposure clipping read as glare, busy surfaces, phone sharpening).

---

## Caveats

- **Detection needs a darker surface.** The page is found by being brighter than what it lies on.
  A white page on a white desk, or a dark card on a light one, is rejected as not found.
- **Rotation is not scored**, and a page turned 90° or 180° is not detected; the top edge's angle is
  reported in `framing.details.rotation_deg` for the downstream pipeline.
- **Glare zone rule not implemented.** For cheques, glare over the amount, the CMC7 line or the
  signature should reject at any size; that needs the zone layout of the cheques you accept.
- **PDF scores describe the render.** A digital-born PDF renders perfectly sharp and glare-free;
  treat its scores as saying nothing about capture quality.
- **Only page 1 of a PDF is scored.** `page_count` in the response tells you what was skipped.
- **`opencv-python` pulls in GUI libraries.** For server deployments, `opencv-python-headless` is a
  drop-in replacement with a smaller footprint and no `libgl1` dependency.

---

## Tests

```bash
pip install -e ".[dev]"
pytest
```

Each check has a standalone test that generates a clean synthetic page, applies one controlled
defect, and asserts the score moves the right way. Pipeline and API tests simulate phone photos —
the page laid on a darker surface, with sensor noise, perspective, cropping — and assert each
defect is blamed on the right check. No sample captures are committed.

```
tests/
├── fixtures.py        synthetic pages, photo simulation + degradations
├── test_document.py   detection and rectification
├── test_framing.py    test_sharpness.py  test_shadows.py  test_glare.py  test_contrast.py
├── test_pipeline.py   weakest-link gate, blame, metadata
└── test_api.py        validation, error shapes, end-to-end decisions
```

---

## Layout

```
src/scan_quality/
├── loaders.py            bytes -> BGR array (PDF page 1 via pdf2image)
├── scoring/              pure functions: array in, CheckResult out. No FastAPI.
│   ├── base.py           CheckResult, labels, ramps, ink mask, background estimate
│   ├── document.py       step 0: find the document, flatten it to physical size
│   ├── framing.py  sharpness.py  shadows.py  glare.py  contrast.py
│   └── pipeline.py       detection, all checks, weakest-link decision
└── api/
    ├── main.py           FastAPI app, CORS, /score, /health
    └── schemas.py        Pydantic response models
```

The `scoring/` package imports nothing from FastAPI and touches no files:

```python
import cv2
from scan_quality.scoring.pipeline import score_image

report = score_image(cv2.imread("claim.jpg"), doc_type="a4")
print(report.accepted, report.retake_message)
```

---

## Production checklist

- **CORS is wide open** (`allow_origins=["*"]`) for local development. `src/scan_quality/api/main.py`
  carries a `TODO` at the middleware — restrict origins and methods before deploying anywhere shared.
- Upload cap is 20 MB (`MAX_UPLOAD_BYTES`); the file is buffered in memory, so size the container
  accordingly for your expected concurrency.
- Scoring takes ~0.5 s for a 2 000 × 2 850 px photo on a laptop CPU and runs in a threadpool, so one
  large upload does not stall other requests. Running on-device (including on live preview frames)
  needs a port to the mobile OpenCV SDK or native code.
