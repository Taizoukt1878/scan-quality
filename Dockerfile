FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System packages:
#   tesseract-ocr      - the OCR engine behind check 5
#   tesseract-ocr-fra  - French language pack (documents in scope are largely
#                        French); drop it if you only ever score English
#   poppler-utils      - pdftoppm/pdfinfo, which pdf2image shells out to
#   libgl1, libglib2.0-0 - shared libraries opencv-python links against
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-fra \
        poppler-utils \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so editing source does not invalidate the wheel cache.
COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-deps .

# Drop privileges: the service only ever reads request bodies into memory.
RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 8000

# Uses the service's own /health endpoint. Written in Python because the image
# ships neither curl nor wget.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "scan_quality.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
