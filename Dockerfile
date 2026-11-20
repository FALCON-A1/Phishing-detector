# Lightweight Python base image
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies (ultra-fast, ~15 seconds)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application and trained model artifact
# Note: datasets are excluded via .dockerignore
COPY feature_extractor.py .
COPY spam_detector.py .
COPY app.py .
COPY index.html .
COPY model.joblib .
COPY spam_model.joblib .

# Expose ports
EXPOSE 8000
EXPOSE 8080

# Start production server binding to 0.0.0.0 and dynamically using Railway's PORT env var
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
