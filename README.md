# URL Phishing Detector

A high-performance, lightweight, and explainable Machine Learning URL Phishing Detection system built with FastAPI and Scikit-Learn.

The system analyzes lexical, structural, security, and typosquatting characteristics of any raw URL and classifies it into one of three categories:
- **Safe**
- **Suspicious**
- **Phishing**

---

## Features

- **32 URL-Based Features**: Evaluates length, entropy, digit ratios, IP usage, character repetition, token distributions, and suspicious keywords.
- **Typosquatting & Lookalike Detection**: Detects character substitutions (e.g., `0 -> o`, `1 -> l`, `rn -> m`), homoglyphs, punycode (`xn--`), and brand impersonation in subdomains.
- **3-Level Decision Logic**: Prevents aggressive false positives on unknown domains while accurately flagging true phishing and lookalike attempts.
- **Explainable Predictions**: Returns human-readable reasons explaining why a URL was marked Safe, Suspicious, or Phishing.
- **Sub-Millisecond Inference**: Pure Python feature extractor running at >3,000 URLs/second.
- **Railway & Docker Ready**: Containerized with `python:3.11-slim`, auto-binding to Railway's `$PORT`.

---

## API Endpoints

### 1. `GET /`
Returns API metadata and documentation links.

### 2. `GET /health`
Returns service status, loaded model type, feature count, and calibrated classification thresholds.

### 3. `POST /predict`
Analyzes a URL and returns its classification, confidence, risk score, and supporting reasons.

#### Request Example
```json
{
  "url": "https://go0g1e.com/login"
}
```

#### Response Example
```json
{
  "url": "https://go0g1e.com/login",
  "prediction": "Phishing",
  "confidence": 0.916,
  "risk_score": 0.916,
  "reasons": [
    "Typosquatting/lookalike domain pattern targeting a well-known brand",
    "Contains 1 suspicious security/login keywords",
    "Model phishing probability is high (91.6%)"
  ]
}
```

---

## Local Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the API
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```
Interactive Swagger documentation is available at `http://localhost:8000/docs`.

---

## Retraining the Model

To retrain the Random Forest model from raw datasets:
```bash
python train.py --data-dir Datasets --output-model model.joblib
```

---

## Railway Deployment

1. Create a new project in [Railway](https://railway.app/).
2. Select **Deploy from GitHub repo** and choose `FALCON-A1/Phishing-detector`.
3. Railway will automatically detect the `Dockerfile`, build the image, and bind to the dynamic `$PORT`.
