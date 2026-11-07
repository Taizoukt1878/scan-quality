# Scan Quality API

Scores how well a document has been scanned — cheques, certificates, devis/quotes — and returns a
0-100 breakdown so you can reject a bad capture before it reaches OCR or a human reviewer.

Everything runs locally on free, open-source software. No external APIs, no network calls, no keys.

Upload a JPEG, PNG or PDF; get back five independent check scores, their unweighted average, a
plain-language label for each, and the total processing time.

---

## The five checks

Each check is a pure function of the pixel data, normalized to 0-100 where **higher is better**.
The overall score is their **simple unweighted average**.

| # | Check | Raw metric | Scores 0 at | Scores 100 at | Module |
|---|-------|------------|-------------|---------------|--------|
| 1 | **Sharpness** | Variance of the Laplacian | `15.0` | `600.0` (log ramp) | `scoring/sharpness.py` |
| 2 | **Lighting uniformity** | Std-dev of mean brightness across an 8×8 grid | `45.0` | `4.0` | `scoring/lighting.py` |
| 3 | **Contrast** | RMS contrast (std-dev of intensity) | `12.0` | `65.0` | `scoring/contrast.py` |
| 4 | **Skew** | Dominant text-line angle, degrees | `5.0°` | `0.5°` | `scoring/skew.py` |
| 5 | **OCR confidence** | Mean per-word Tesseract confidence | `40.0` | `92.0` | `scoring/ocr.py` |

Checks 2 and 4 run *backwards* — less spread and less tilt are better — which is why their "0 at"
value is the larger one.

### Labels

The same vocabulary applies to every check and to the overall score:

| Score | Label |
|-------|-------|
| ≥ 85 | `excellent` |
| ≥ 70 | `good` |
| ≥ 50 | `fair` |
| < 50 | `poor` |

A check that could not measure anything at all reports `unknown` (see *Caveats*).

### Notes on the methods

**Sharpness** uses a logarithmic ramp because Laplacian variance spans orders of magnitude — the
perceptual gap between 20 and 60 is far larger than between 500 and 540. The image is first
resampled to a fixed working size, because the same physical blur measures much higher on a 4000px
scan than a 1000px one.

**Skew** runs the Hough Transform over *character baseline points* — each glyph reduced to one pixel
at the centre of its bottom edge — rather than over text pixels or word blobs. This matters: any
line running lengthwise through a blob of height `h` and length `L` collects a full set of votes, so
every angle within `±atan(h/L)` ties. For body text that ambiguity cone exceeds a degree, which is
larger than the whole range this check cares about. One-pixel points have no such cone, and the
estimate lands within ~0.13° of truth. Binarization is adaptive, so skew stays measurable on pages
that fail the blur and lighting checks.

---

## Prerequisites

Two **system** packages are required in addition to the Python dependencies. They are not pip
packages — `pytesseract` and `pdf2image` are thin wrappers that shell out to these binaries.

| Binary | Needed for | Without it |
|--------|-----------|------------|
| `tesseract` | Check 5 (OCR confidence) | `/score` returns **503** for every request |
| `poppler` (`pdftoppm`, `pdfinfo`) | Rendering PDF uploads | PDF uploads return **503**; JPEG/PNG still work |

**Ubuntu / Debian**

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr poppler-utils

# Optional: French language pack, for devis/quotes and certificates
sudo apt-get install -y tesseract-ocr-fra
```

**macOS (Homebrew)**

```bash
brew install tesseract poppler

# Optional: all language packs, including French
brew install tesseract-lang
```

Verify:

```bash
tesseract --version
pdftoppm -v
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

The image installs Tesseract (with the French pack) and poppler for you.

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
| `file` | form-data | yes | The document. For a PDF, **only page 1 is scored**. |
| `lang` | query | no | Tesseract language pack(s), e.g. `fra` or `fra+eng`. Default `eng`. Must be installed — the service checks up front and returns a 400 naming what is missing and what is available. |

```bash
curl -X POST http://127.0.0.1:8000/score \
  -F "file=@devis.jpg" \
  -F "lang=fra"   # or: ?lang=fra
```

The file type is decided by the file's **magic bytes**, not its extension or declared content type —
a PNG named `.pdf` is scored correctly as a PNG.

**Example response**

```json
{
  "overall_score": 90.33,
  "overall_label": "excellent",
  "checks": [
    {
      "name": "sharpness",
      "score": 100.0,
      "label": "excellent",
      "raw_value": 1500.31,
      "raw_metric": "Laplacian variance",
      "details": { "working_max_dim": 1500, "duration_ms": 8.85 }
    },
    {
      "name": "lighting_uniformity",
      "score": 74.96,
      "label": "good",
      "raw_value": 14.27,
      "raw_metric": "std-dev of per-cell mean brightness (0-255)",
      "details": { "grid": "8x8", "darkest_cell": 206.26, "brightest_cell": 245.0, "duration_ms": 1.57 }
    },
    {
      "name": "contrast",
      "score": 76.71,
      "label": "good",
      "raw_value": 52.66,
      "raw_metric": "RMS contrast (std-dev of intensity, 0-255)",
      "details": { "duration_ms": 4.78 }
    },
    {
      "name": "skew",
      "score": 100.0,
      "label": "excellent",
      "raw_value": 0.0,
      "raw_metric": "dominant text-line angle (degrees from horizontal)",
      "details": { "text_lines_detected": 172, "baseline_points": 980, "duration_ms": 23.81 }
    },
    {
      "name": "ocr_confidence",
      "score": 100.0,
      "label": "excellent",
      "raw_value": 92.98,
      "raw_metric": "mean per-word Tesseract confidence (0-100)",
      "details": { "word_count": 196, "lang": "eng", "duration_ms": 637.86 }
    }
  ],
  "processing_time_ms": 681.75,
  "file": {
    "filename": "devis.jpg",
    "size_bytes": 332943,
    "source_format": "jpeg",
    "page_count": 1,
    "scored_page": 1,
    "width": 1200,
    "height": 1600,
    "render_dpi": null
  }
}
```

Note the timing split, measured on a 1200×1600 page: OCR took 638 ms of the 682 ms total. The four
pixel-statistics checks together cost under 40 ms. If you need a fast pre-filter, run those four and
skip check 5.

`raw_value` is the measurement *before* normalization. It is in the response on purpose: it is what
you need to re-tune the thresholds against your own scans without re-instrumenting the service.

### `GET /health`

```json
{ "status": "ok", "version": "0.1.0", "tesseract_available": true, "tesseract_version": "5.3.4" }
```

`tesseract_available: false` means every `/score` call will fail — worth alerting on.

### Errors

All errors share one shape:

```json
{ "error": { "code": "unsupported_file_type", "message": "File is not a JPEG, PNG or PDF ..." } }
```

| Status | `code` | Cause |
|--------|--------|-------|
| 400 | `corrupt_file` | Empty, truncated or undecodable file |
| 400 | `invalid_language` | `lang` is not of the form `xxx` or `xxx+yyy` |
| 400 | `language_not_available` | `lang` is well-formed but that pack is not installed |
| 413 | `file_too_large` | Upload exceeds 20 MB |
| 415 | `unsupported_file_type` | Contents are not JPEG, PNG or PDF |
| 422 | — | Missing `file` field (FastAPI's own validation) |
| 503 | `ocr_unavailable` | Tesseract binary not found |
| 503 | `pdf_render_unavailable` | poppler not found |

A missing binary is deliberately a 503 rather than a low score: a deployment fault must not be
reported as a bad scan.

---

## Tuning

Every threshold is a named constant at the top of the check's own module, with a comment explaining
the choice. To recalibrate, collect real scans, read the `raw_value` fields off the responses, and
move the anchors.

The defaults are reasoned starting points, not measurements from your corpus. Expect to adjust:

- `LOW_CONTRAST_FLOOR` / `HIGH_CONTRAST_CEILING` in `contrast.py` — sparse documents (a cheque, a
  short quote) are mostly white paper and measure lower than a dense page, without being worse.
- `UNIFORM_STD_FLOOR` / `UNEVEN_STD_CEILING` in `lighting.py` — see the caveat below.
- `OCR_CONF_FLOOR` / `OCR_CONF_CEILING` in `ocr.py` — the usable band shifts with the language pack
  and the document's vocabulary.

---

## Caveats

- **The lighting check also responds to layout.** Cells over dense text average darker than cells
  over margins, so a page with wide white margins registers some spread even under perfectly even
  light. On the clean synthetic test page the check scores ~75, not 100. Calibrate the constants
  against your own layouts rather than assuming 100 is reachable.
- **Skew can report `unknown`.** With too few characters to fit baselines to — a near-blank page, a
  cheque stub — the check returns a neutral score of 50 and the label `unknown` rather than guessing.
  It contributes to the average either way, so a mostly-blank page is neither rewarded nor punished.
- **PDF scores describe the render, not a scan.** A digital-born PDF rasterizes perfectly sharp and
  perfectly straight, so checks 1 and 4 will report near-perfect scores that say nothing about
  capture quality. This service is aimed at *scanned* pages; treat scores from vector PDFs with
  suspicion.
- **Only page 1 of a PDF is scored.** `page_count` in the response tells you what was skipped.
- **Skew is small-angle only.** A page rotated 90° or upside down is not detected; angles beyond
  ±20° are ignored as non-text.
- **`opencv-python` pulls in GUI libraries.** For server deployments, `opencv-python-headless` is a
  drop-in replacement with a smaller footprint and no `libgl1` dependency.

---

## Tests

```bash
pip install -e ".[dev]"
pytest
```

Each check has a standalone test that generates a clean synthetic page, applies one controlled
defect, and asserts the score moves the right way — including that known rotations are recovered to
within 0.5°. No sample scans are committed.

Tests that need the Tesseract binary skip cleanly when it is absent, so the suite runs anywhere. The
API tests stub only `image_to_data`, so the HTTP layer and the other four checks are always
exercised for real.

```
tests/
├── fixtures.py       synthetic pages + degradations (blur, tilt, gradient, wash-out)
├── test_sharpness.py test_lighting.py test_contrast.py test_skew.py test_ocr.py
├── test_pipeline.py  averaging, ordering, timing
└── test_api.py       validation, error shapes, end-to-end scoring
```

---

## Layout

```
src/scan_quality/
├── loaders.py            bytes -> BGR array (PDF page 1 via pdf2image)
├── scoring/              pure functions: array in, CheckResult out. No FastAPI.
│   ├── base.py           CheckResult, label thresholds, normalization helpers
│   ├── sharpness.py  lighting.py  contrast.py  skew.py  ocr.py
│   └── pipeline.py       runs all five, averages, times
└── api/
    ├── main.py           FastAPI app, CORS, /score, /health
    └── schemas.py        Pydantic response models
```

The `scoring/` package imports nothing from FastAPI and touches no files, so every check can be
called directly on a NumPy array:

```python
import cv2
from scan_quality.scoring.sharpness import score_sharpness

print(score_sharpness(cv2.imread("devis.jpg")))
```

---

## Production checklist

- **CORS is wide open** (`allow_origins=["*"]`) for local development. `src/scan_quality/api/main.py`
  carries a `TODO` at the middleware — restrict origins and methods before deploying anywhere shared.
- Upload cap is 20 MB (`MAX_UPLOAD_BYTES`); the file is buffered in memory, so size the container
  accordingly for your expected concurrency.
- Scoring runs in a threadpool, so a slow OCR pass will not stall other in-flight requests. OCR
  dominates the runtime — roughly an order of magnitude more than the four pixel-statistics checks.
