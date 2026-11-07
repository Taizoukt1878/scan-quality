"""HTTP layer: validation, error shapes and the happy path.

The happy-path tests stub Tesseract's ``image_to_data`` rather than skipping,
so the full pipeline and response serialization are exercised even on a
machine without the binary installed. The stub is the only thing faked — all
five checks run for real on real pixels.
"""

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from scan_quality.api.main import MAX_UPLOAD_BYTES, app
from tests import fixtures

EXPECTED_CHECKS = {"sharpness", "lighting_uniformity", "contrast", "skew", "ocr_confidence"}


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def stub_tesseract(monkeypatch):
    """Make the OCR check return a fixed, plausible confidence."""
    import scan_quality.scoring.ocr as ocr_module

    # Also bypass the language-pack check, so these tests keep running on a
    # machine that has no Tesseract installed at all.
    monkeypatch.setattr(ocr_module, "_require_languages", lambda lang: None)

    def fake_image_to_data(image, **kwargs):
        return {
            "conf": ["-1", "88", "91", "76", "84"],
            "text": ["", "Devis", "numero", "Total", "EUR"],
        }

    monkeypatch.setattr(ocr_module.pytesseract, "image_to_data", fake_image_to_data)


def _png_bytes(image: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    return buffer.tobytes()


def _jpeg_bytes(image: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    assert ok
    return buffer.tobytes()


def test_health_reports_status_and_version(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert isinstance(body["tesseract_available"], bool)


def test_score_returns_all_five_checks(client, stub_tesseract):
    payload = _png_bytes(fixtures.clean_page())
    response = client.post("/score", files={"file": ("devis.png", payload, "image/png")})

    assert response.status_code == 200, response.text
    body = response.json()
    assert {check["name"] for check in body["checks"]} == EXPECTED_CHECKS
    assert 0 <= body["overall_score"] <= 100
    assert body["overall_label"] in {"excellent", "good", "fair", "poor"}


def test_overall_score_is_the_mean_of_the_five(client, stub_tesseract):
    payload = _png_bytes(fixtures.clean_page())
    body = client.post("/score", files={"file": ("d.png", payload, "image/png")}).json()
    mean = sum(check["score"] for check in body["checks"]) / 5
    assert body["overall_score"] == pytest.approx(mean, abs=0.01)


def test_response_reports_file_metadata_and_timing(client, stub_tesseract):
    page = fixtures.clean_page(width=800, height=1000)
    payload = _jpeg_bytes(page)
    body = client.post("/score", files={"file": ("scan.jpg", payload, "image/jpeg")}).json()

    assert body["file"]["filename"] == "scan.jpg"
    assert body["file"]["source_format"] == "jpeg"
    assert body["file"]["size_bytes"] == len(payload)
    assert body["file"]["width"] == 800
    assert body["file"]["height"] == 1000
    assert body["file"]["scored_page"] == 1
    assert body["processing_time_ms"] > 0


def test_a_bad_scan_scores_below_a_clean_one(client, stub_tesseract):
    """End-to-end sanity: the API ranks scans the way a human would."""
    clean = fixtures.clean_page()
    ruined = fixtures.low_contrast(
        fixtures.uneven_lighting(fixtures.blurred(fixtures.rotated(clean, 4.0), 15), 0.7), 0.4
    )
    scores = [
        client.post("/score", files={"file": ("x.png", _png_bytes(image), "image/png")}).json()[
            "overall_score"
        ]
        for image in (clean, ruined)
    ]
    assert scores[0] > scores[1]


def test_file_type_is_detected_from_content_not_extension(client, stub_tesseract):
    """A PNG named .pdf with an image content type is still scored as a PNG."""
    payload = _png_bytes(fixtures.clean_page())
    response = client.post("/score", files={"file": ("misnamed.pdf", payload, "application/pdf")})
    assert response.status_code == 200
    assert response.json()["file"]["source_format"] == "png"


def test_unsupported_file_type_is_rejected(client):
    response = client.post("/score", files={"file": ("notes.txt", b"just some text", "text/plain")})
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_file_type"


def test_empty_file_is_rejected(client):
    response = client.post("/score", files={"file": ("empty.png", b"", "image/png")})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "corrupt_file"


def test_truncated_image_is_rejected(client):
    """Correct magic bytes, unusable contents."""
    truncated = _png_bytes(fixtures.clean_page())[:120]
    response = client.post("/score", files={"file": ("cut.png", truncated, "image/png")})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "corrupt_file"


def test_oversized_upload_is_rejected(client):
    oversized = b"\x89PNG\r\n\x1a\n" + b"\x00" * MAX_UPLOAD_BYTES
    response = client.post("/score", files={"file": ("big.png", oversized, "image/png")})
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "file_too_large"


@pytest.mark.parametrize("lang", ["../etc", "en", "fra;rm -rf", "FRA"])
def test_invalid_language_is_rejected(client, lang):
    payload = _png_bytes(fixtures.clean_page())
    response = client.post(
        "/score", files={"file": ("d.png", payload, "image/png")}, params={"lang": lang}
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_language"


def test_uninstalled_language_returns_400_not_500(client):
    """Asking for a pack the server lacks must be a clear error, not a crash.

    Tesseract's own failure mode here is an opaque message about
    TESSDATA_PREFIX, so the service checks the packs up front instead.
    """
    from scan_quality.scoring.ocr import available_languages, is_tesseract_available

    if not is_tesseract_available():
        pytest.skip("Tesseract binary not installed")
    if "zzz" in available_languages():
        pytest.skip("improbable language pack is actually installed")

    payload = _png_bytes(fixtures.clean_page())
    response = client.post(
        "/score", files={"file": ("d.png", payload, "image/png")}, params={"lang": "zzz"}
    )
    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == "language_not_available"
    # The message must name what is missing and what is available.
    assert "zzz" in body["error"]["message"]
    assert "Available:" in body["error"]["message"]


def test_unexpected_tesseract_failure_returns_503(client, monkeypatch):
    """Engine errors must be wrapped, not escape as an unhandled 500."""
    from pytesseract import TesseractError

    import scan_quality.scoring.ocr as ocr_module

    monkeypatch.setattr(ocr_module, "_require_languages", lambda lang: None)

    def explode(*args, **kwargs):
        raise TesseractError(1, "corrupt traineddata")

    monkeypatch.setattr(ocr_module.pytesseract, "image_to_data", explode)
    payload = _png_bytes(fixtures.clean_page())
    response = client.post("/score", files={"file": ("d.png", payload, "image/png")})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "ocr_unavailable"


def test_missing_file_field_is_rejected(client):
    assert client.post("/score").status_code == 422


def test_missing_tesseract_returns_503(client, monkeypatch):
    from pytesseract import TesseractNotFoundError

    import scan_quality.scoring.ocr as ocr_module

    def explode(*args, **kwargs):
        raise TesseractNotFoundError()

    monkeypatch.setattr(ocr_module.pytesseract, "image_to_data", explode)
    payload = _png_bytes(fixtures.clean_page())
    response = client.post("/score", files={"file": ("d.png", payload, "image/png")})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "ocr_unavailable"


def _pdf_bytes(image: np.ndarray) -> bytes:
    """Wrap an image in a single-page PDF using Pillow's PDF writer."""
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).save(buffer, format="PDF")
    return buffer.getvalue()


@pytest.mark.skipif(
    __import__("shutil").which("pdftoppm") is None, reason="poppler not installed"
)
def test_pdf_first_page_is_scored(client, stub_tesseract):
    payload = _pdf_bytes(fixtures.clean_page())
    response = client.post("/score", files={"file": ("devis.pdf", payload, "application/pdf")})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["file"]["source_format"] == "pdf"
    assert body["file"]["scored_page"] == 1
    assert body["file"]["render_dpi"] == 200
    assert {check["name"] for check in body["checks"]} == EXPECTED_CHECKS


@pytest.mark.skipif(
    __import__("shutil").which("pdftoppm") is not None, reason="poppler is installed"
)
def test_pdf_without_poppler_returns_503(client):
    payload = _pdf_bytes(fixtures.clean_page())
    response = client.post("/score", files={"file": ("devis.pdf", payload, "application/pdf")})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "pdf_render_unavailable"


def test_unknown_route_uses_the_shared_error_shape(client):
    response = client.get("/nope")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_cors_headers_are_open_for_local_dev(client):
    response = client.get("/health", headers={"Origin": "http://localhost:3000"})
    assert response.headers["access-control-allow-origin"] == "*"
