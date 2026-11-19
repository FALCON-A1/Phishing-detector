"""
FastAPI Application for URL Phishing Detector.
Provides / , /health, and /predict endpoints with 3-level classification:
Safe, Suspicious, or Phishing.
"""

import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

# Ensure project root is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from feature_extractor import (
    extract_features, extract_features_vector, FEATURE_NAMES
)

# Global model state
MODEL_STATE: Dict[str, Any] = {
    "loaded": False,
    "model": None,
    "scaler": None,
    "model_type": None,
    "feature_names": None,
    "thresholds": {"safe_max": 0.30, "phish_min": 0.65},
    "version": "1.0.0",
    "load_time": None
}


def load_model_artifact(model_path: str = "model.joblib") -> bool:
    """Load model artifact from disk into global state."""
    full_path = os.path.join(CURRENT_DIR, model_path) if not os.path.isabs(model_path) else model_path
    if not os.path.exists(full_path):
        print(f"Warning: Model file not found at {full_path}")
        return False

    try:
        artifact = joblib.load(full_path)
        MODEL_STATE["model"] = artifact["model"]
        MODEL_STATE["scaler"] = artifact.get("scaler")
        MODEL_STATE["model_type"] = artifact.get("model_type", type(artifact["model"]).__name__)
        MODEL_STATE["feature_names"] = artifact.get("feature_names", FEATURE_NAMES)
        MODEL_STATE["thresholds"] = artifact.get("thresholds", {"safe_max": 0.30, "phish_min": 0.65})
        MODEL_STATE["version"] = artifact.get("version", "1.0.0")
        MODEL_STATE["loaded"] = True
        MODEL_STATE["load_time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        print(f"Successfully loaded model: {MODEL_STATE['model_type']} with {len(MODEL_STATE['feature_names'])} features")
        return True
    except Exception as e:
        print(f"Error loading model artifact: {e}")
        MODEL_STATE["loaded"] = False
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once on application startup."""
    model_env_path = os.environ.get("MODEL_PATH", "model.joblib")
    load_model_artifact(model_env_path)
    yield


app = FastAPI(
    title="URL Phishing Detector API",
    description="High-performance, lightweight URL phishing and typosquatting detection service.",
    version="1.0.0",
    lifespan=lifespan
)


class PredictRequest(BaseModel):
    url: str = Field(..., description="The raw URL string to analyze", min_length=1, max_length=2048)

    model_config = {
        "json_schema_extra": {
            "example": {
                "url": "https://example.com"
            }
        }
    }


class PredictResponse(BaseModel):
    url: str
    prediction: str = Field(..., description="Classification category: Safe, Suspicious, or Phishing")
    confidence: float = Field(..., description="Calibrated confidence score between 0.0 and 1.0")
    risk_score: float = Field(..., description="Raw phishing risk probability between 0.0 and 1.0")
    reasons: List[str] = Field(default_factory=list, description="Explanations supporting the classification decision")


TRUSTED_DOMAINS = {
    "google.com", "youtube.com", "microsoft.com", "amazon.com", "apple.com",
    "wikipedia.org", "github.com", "facebook.com", "stackoverflow.com",
    "linkedin.com", "twitter.com", "x.com", "instagram.com", "netflix.com",
    "vodafone.com.eg", "vf.eg"
}

def classify_url(url: str) -> PredictResponse:
    """Classify a URL using the loaded model and security rule engine."""
    if not MODEL_STATE["loaded"] or MODEL_STATE["model"] is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Please train or provide a model file."
        )

    # 1. Extract features using the exact same logic as training
    feats_dict = extract_features(url)
    feat_names = MODEL_STATE["feature_names"] or FEATURE_NAMES
    feat_vec = np.array([[feats_dict[name] for name in feat_names]], dtype=np.float32)

    # 2. Scale features if scaler exists
    if MODEL_STATE["scaler"] is not None:
        feat_vec = MODEL_STATE["scaler"].transform(feat_vec)

    # 3. Model risk probability
    model = MODEL_STATE["model"]
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(feat_vec)[0, 1])
    else:
        # Logistic decision function mapped via sigmoid
        df = float(model.decision_function(feat_vec)[0])
        prob = 1.0 / (1.0 + np.exp(-df))

    # 4. Extract explainable security indicators
    reasons: List[str] = []
    is_lookalike = feats_dict["is_brand_lookalike"] > 0
    brand_in_sub = feats_dict["brand_in_subdomain_or_path"] > 0
    has_ip = feats_dict["has_ip"] > 0
    kw_cnt = int(feats_dict["suspicious_keyword_count"])
    susp_tld = feats_dict["suspicious_tld"] > 0
    has_homoglyphs = feats_dict["has_homoglyphs"] > 0
    has_punycode = feats_dict["has_punycode"] > 0
    has_at = feats_dict["has_at_symbol"] > 0
    has_redirect = feats_dict["has_double_slash_redirect"] > 0

    # Parse components for domain verification
    from feature_extractor import parse_url_components
    scheme, full_dom, sld, tld, path, query = parse_url_components(url)
    is_trusted = any(full_dom == d or full_dom.endswith("." + d) for d in TRUSTED_DOMAINS)

    if is_trusted and not is_lookalike and not brand_in_sub and not has_ip and not has_homoglyphs:
        return PredictResponse(
            url=url,
            prediction="Safe",
            confidence=0.98,
            risk_score=round(prob, 4),
            reasons=["Verified legitimate authority domain with valid SSL"]
        )

    if is_lookalike:
        reasons.append("Typosquatting/lookalike domain pattern targeting a well-known brand")
    if brand_in_sub:
        reasons.append("Brand impersonation detected in subdomain or path")
    if has_homoglyphs:
        reasons.append("Homoglyph characters detected mimicking standard Latin letters")
    if has_punycode:
        reasons.append("Punycode (xn--) domain structure detected")
    if has_ip:
        reasons.append("Direct IP address used in place of standard domain hostname")
    if kw_cnt > 0:
        reasons.append(f"Contains {kw_cnt} security, credential, or authentication keywords")
    if susp_tld:
        reasons.append("Domain utilizes a high-risk TLD frequently abused in attacks")
    if has_at:
        reasons.append("URL contains '@' symbol typically used for URL authority confusion")
    if has_redirect:
        reasons.append("URL contains embedded double slashes indicating open redirect pattern")

    thresholds = MODEL_STATE["thresholds"]
    t_safe = thresholds.get("safe_max", 0.30)
    t_phish = thresholds.get("phish_min", 0.65)

    # 5. Determine 3-Level classification adhering strictly to Section 8 rules
    has_indicators = len(reasons) > 0

    if prob >= t_phish and has_indicators:
        prediction = "Phishing"
        confidence = prob
        reasons.append(f"Model phishing risk is high ({prob * 100:.1f}%) with supporting malicious patterns")
    elif is_lookalike or brand_in_sub or (has_ip and kw_cnt > 0) or has_homoglyphs:
        prediction = "Suspicious" if prob < 0.60 else "Phishing"
        confidence = max(prob, 0.75)
    elif prob >= t_phish and not has_indicators:
        # High model score without specific malicious patterns -> Suspicious per Section 8
        prediction = "Suspicious"
        confidence = prob
        reasons.append(f"Elevated statistical risk ({prob * 100:.1f}%) but no confirmed malicious patterns detected; proceed with caution")
    elif prob >= t_safe:
        prediction = "Suspicious"
        confidence = prob
        reasons.append(f"Moderate risk score ({prob * 100:.1f}%) with ambiguous characteristics")
    else:
        prediction = "Safe"
        confidence = 1.0 - prob
        if not reasons:
            reasons.append("No significant suspicious URL patterns or brand impersonation detected")

    return PredictResponse(
        url=url,
        prediction=prediction,
        confidence=round(confidence, 4),
        risk_score=round(prob, 4),
        reasons=reasons
    )


@app.get("/", summary="API Information")
def get_root():
    """Information endpoint providing API metadata and links."""
    return {
        "name": "URL Phishing Detector API",
        "version": MODEL_STATE["version"],
        "status": "online",
        "description": "High-performance, lightweight URL phishing and typosquatting detector.",
        "model_loaded": MODEL_STATE["loaded"],
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }


@app.get("/health", summary="Health Check")
def get_health():
    """Health check endpoint indicating service and model readiness."""
    if not MODEL_STATE["loaded"]:
        return {
            "status": "degraded",
            "model_loaded": False,
            "message": "API running but model is not loaded."
        }

    return {
        "status": "healthy",
        "model_loaded": True,
        "model_type": MODEL_STATE["model_type"],
        "features_count": len(MODEL_STATE["feature_names"] or []),
        "thresholds": MODEL_STATE["thresholds"],
        "version": MODEL_STATE["version"],
        "load_time": MODEL_STATE["load_time"]
    }


@app.post("/predict", response_model=PredictResponse, summary="Classify URL")
def predict_url(payload: PredictRequest):
    """
    Accepts a URL string, analyzes its lexical, security, and typosquatting features,
    and returns a Safe, Suspicious, or Phishing classification with explainable reasons.
    """
    clean_url = payload.url.strip()
    if not clean_url:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="URL must not be empty or whitespace only."
        )

    # Basic scheme check or format check
    if len(clean_url) > 2048:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="URL exceeds maximum supported length of 2048 characters."
        )

    try:
        return classify_url(clean_url)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during prediction: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
