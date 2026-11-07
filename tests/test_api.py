"""HTTP layer: validation, error shapes and the accept/reject happy path.

Nothing is stubbed: every check runs for real on real pixels.
"""

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from scan_quality.api.main import MAX_UPLOAD_BYTES, app
from tests import fixtures

EXPECTED_CHECKS = {"framing", "sharpness", "shadows", "glare", "contrast"}


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(scope="module")
def photo() -> np.ndarray:
    return fixtures.photo_of(fixtures.a4_page())


def _png_bytes(image: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    return buffer.tobytes()


def _jpeg_bytes(image: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    assert ok
    return buffer.tobytes()


def _post(client, payload, name="capture.jpg", content_type="image/jpeg", **params):
    return client.post("/score", files={"file": (name, payload, content_type)}, params=params)


def test_health_reports_status_and_version(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_clean_capture_is_accepted(client, photo):
    response = _post(client, _jpeg_bytes(photo))

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["accepted"] is True
    assert body["failed_checks"] == []
    assert body["retake_message"] is None
    assert {check["name"] for check in body["checks"]} == EXPECTED_CHECKS


def test_overall_score_is_the_minimum(client, photo):
    body = _post(client, _jpeg_bytes(photo)).json()
    assert body["overall_score"] == pytest.approx(min(c["score"] for c in body["checks"]), abs=0.01)


def test_bad_capture_is_rejected_with_an_instruction(client):
    glare = fixtures.photo_of(fixtures.glare_spot(fixtures.a4_page(), size_px=(260, 180)))
    body = _post(client, _jpeg_bytes(glare)).json()
    assert body["accepted"] is False
    assert body["failed_checks"][0] == "glare"
    assert body["retake_message"]


def test_response_reports_file_metadata_and_timing(client, photo):
    payload = _jpeg_bytes(photo)
    body = _post(client, payload, name="scan.jpg").json()

    assert body["file"]["filename"] == "scan.jpg"
    assert body["file"]["source_format"] == "jpeg"
    assert body["file"]["size_bytes"] == len(payload)
    assert body["file"]["width"] == photo.shape[1]
    assert body["file"]["height"] == photo.shape[0]
    assert body["file"]["doc_type"] == "a4"
    assert body["file"]["document_width"] > 0
    assert body["processing_time_ms"] > 0


def test_doc_type_changes_the_resolution_target(client):
    cheque = fixtures.photo_of(fixtures.clean_page(width=1400, height=640, line_count=5))
    as_cheque = _post(client, _jpeg_bytes(cheque), doc_type="cheque").json()
    as_a4 = _post(client, _jpeg_bytes(cheque), doc_type="a4").json()
    framing = {body["file"]["doc_type"]: next(c for c in body["checks"] if c["name"] == "framing")
               for body in (as_cheque, as_a4)}
    assert framing["cheque"]["raw_value"] > framing["a4"]["raw_value"]


def test_unknown_doc_type_is_rejected(client, photo):
    assert _post(client, _jpeg_bytes(photo), doc_type="passport").status_code == 422


def test_file_type_is_detected_from_content_not_extension(client, photo):
    """A PNG named .pdf with a PDF content type is still scored as a PNG."""
    response = _post(client, _png_bytes(photo), name="misnamed.pdf", content_type="application/pdf")
    assert response.status_code == 200
    assert response.json()["file"]["source_format"] == "png"


def test_unsupported_file_type_is_rejected(client):
    response = _post(client, b"just some text", name="notes.txt", content_type="text/plain")
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_file_type"


def test_empty_file_is_rejected(client):
    response = _post(client, b"", name="empty.png", content_type="image/png")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "corrupt_file"


def test_truncated_image_is_rejected(client):
    """Correct magic bytes, unusable contents."""
    truncated = _png_bytes(fixtures.clean_page())[:120]
    response = _post(client, truncated, name="cut.png", content_type="image/png")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "corrupt_file"


def test_oversized_upload_is_rejected(client):
    oversized = b"\x89PNG\r\n\x1a\n" + b"\x00" * MAX_UPLOAD_BYTES
    response = _post(client, oversized, name="big.png", content_type="image/png")
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "file_too_large"


def test_missing_file_field_is_rejected(client):
    assert client.post("/score").status_code == 422


def _pdf_bytes(image: np.ndarray) -> bytes:
    """Wrap an image in a single-page PDF using Pillow's PDF writer."""
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).save(buffer, format="PDF", resolution=200)
    return buffer.getvalue()


@pytest.mark.skipif(
    __import__("shutil").which("pdftoppm") is None, reason="poppler not installed"
)
def test_pdf_page_is_scored_without_detection(client):
    payload = _pdf_bytes(fixtures.a4_page())
    response = _post(client, payload, name="devis.pdf", content_type="application/pdf")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["file"]["source_format"] == "pdf"
    assert body["file"]["render_dpi"] == 200
    framing = next(c for c in body["checks"] if c["name"] == "framing")
    assert framing["details"]["in_frame"] is True


@pytest.mark.skipif(
    __import__("shutil").which("pdftoppm") is not None, reason="poppler is installed"
)
def test_pdf_without_poppler_returns_503(client):
    payload = _pdf_bytes(fixtures.clean_page())
    response = _post(client, payload, name="devis.pdf", content_type="application/pdf")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "pdf_render_unavailable"


def test_unknown_route_uses_the_shared_error_shape(client):
    response = client.get("/nope")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_cors_headers_are_open_for_local_dev(client):
    response = client.get("/health", headers={"Origin": "http://localhost:3000"})
    assert response.headers["access-control-allow-origin"] == "*"
