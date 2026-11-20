"""
FastAPI Application for Security AI Suite:
1. URL Phishing Detector (3-level classification: Safe, Suspicious, Phishing)
2. Multilingual Spam vs Ham Detector for English and Arabic messages
"""

import asyncio
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# Ensure project root is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_HTML_PATH = os.path.join(CURRENT_DIR, "index.html")
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from feature_extractor import (
    extract_features, extract_features_vector, FEATURE_NAMES,
    parse_url_components, detect_brand_lookalike
)
from spam_detector import (
    load_spam_model, classify_message, SPAM_MODEL_STATE
)

# Global model state for URL Phishing Detector
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
    """Load URL model artifact from disk into global state."""
    full_path = os.path.join(CURRENT_DIR, model_path) if not os.path.isabs(model_path) else model_path
    if not os.path.exists(full_path):
        print(f"Warning: URL model file not found at {full_path}")
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
        print(f"Successfully loaded URL model: {MODEL_STATE['model_type']} with {len(MODEL_STATE['feature_names'])} features")
        return True
    except Exception as e:
        print(f"Error loading URL model artifact: {e}")
        MODEL_STATE["loaded"] = False
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML models once on application startup and bridge alternate Railway ports."""
    # 1. Load URL Phishing Model
    model_env_path = os.environ.get("MODEL_PATH", "model.joblib")
    load_model_artifact(model_env_path)

    # 2. Load Spam vs Ham Detector Model
    spam_env_path = os.environ.get("SPAM_MODEL_PATH", "spam_model.joblib")
    load_spam_model(spam_env_path)

    # 3. Railway port bridge: ensure both port 8000 and 8080 respond cleanly
    current_port = int(os.environ.get("PORT", 8000))
    aux_ports = [p for p in (8000, 8080) if p != current_port]
    aux_servers = []

    async def forward_connection(reader, writer, target_port):
        try:
            t_reader, t_writer = await asyncio.open_connection("127.0.0.1", target_port)
            async def pipe(r, w):
                try:
                    while not r.at_eof():
                        data = await r.read(65536)
                        if not data:
                            break
                        w.write(data)
                        await w.drain()
                except Exception:
                    pass
                finally:
                    try:
                        w.close()
                    except Exception:
                        pass
            await asyncio.gather(pipe(reader, t_writer), pipe(t_reader, writer))
        except Exception:
            try:
                writer.close()
            except Exception:
                pass

    for aux_port in aux_ports:
        try:
            srv = await asyncio.start_server(
                lambda r, w, tp=current_port: forward_connection(r, w, tp),
                "0.0.0.0",
                aux_port
            )
            aux_servers.append(srv)
            print(f"Dual-port bridge active: forwarding 0.0.0.0:{aux_port} -> 127.0.0.1:{current_port}")
        except Exception as e:
            print(f"Auxiliary port {aux_port} bridge skipped: {e}")

    yield

    for srv in aux_servers:
        srv.close()


app = FastAPI(
    title="Security AI Protection Suite API",
    description="High-performance, lightweight URL Phishing Detector and Multilingual Spam/Ham Message Detector.",
    version="1.1.0",
    lifespan=lifespan
)


# ==========================================
# 1. URL PHISHING DETECTOR SCHEMAS & LOGIC
# ==========================================

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
    risk_score: float = Field(..., description="Raw or calibrated phishing risk score between 0.0 and 1.0")
    reasons: List[str] = Field(default_factory=list, description="Explanations supporting the classification decision")


TRUSTED_DOMAINS = {
    "google.com", "youtube.com", "microsoft.com", "amazon.com", "apple.com",
    "wikipedia.org", "github.com", "facebook.com", "stackoverflow.com",
    "linkedin.com", "twitter.com", "x.com", "instagram.com", "netflix.com",
    "vodafone.com.eg", "vf.eg", "paypal.com", "yahoo.com", "spotify.com",
    "whatsapp.com", "tiktok.com", "dropbox.com", "ebay.com", "walmart.com"
}


def classify_url(url: str) -> PredictResponse:
    """Classify a URL using the loaded model and security rule engine."""
    if not MODEL_STATE["loaded"] or MODEL_STATE["model"] is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="URL model is not loaded. Please train or provide a model file."
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
        df = float(model.decision_function(feat_vec)[0])
        prob = 1.0 / (1.0 + np.exp(-df))

    # 4. Extract explainable security indicators
    reasons: List[str] = []
    lookalike_info = detect_brand_lookalike(url)
    is_lookalike = (feats_dict["is_brand_lookalike"] > 0) or (lookalike_info is not None)
    target_brand = lookalike_info[0] if lookalike_info else None
    target_token = lookalike_info[1] if lookalike_info else None

    brand_in_sub = feats_dict["brand_in_subdomain_or_path"] > 0
    has_ip = feats_dict["has_ip"] > 0
    kw_cnt = int(feats_dict["suspicious_keyword_count"])
    susp_tld = feats_dict["suspicious_tld"] > 0
    has_homoglyphs = feats_dict["has_homoglyphs"] > 0
    has_punycode = feats_dict["has_punycode"] > 0
    has_at = feats_dict["has_at_symbol"] > 0
    has_redirect = feats_dict["has_double_slash_redirect"] > 0

    # Parse components for domain verification
    scheme, full_dom, sld, tld, path, query = parse_url_components(url)
    is_trusted = any(full_dom == d or full_dom.endswith("." + d) for d in TRUSTED_DOMAINS)

    if is_trusted and not is_lookalike and not brand_in_sub and not has_ip and not has_homoglyphs:
        return PredictResponse(
            url=url,
            prediction="Safe",
            confidence=0.98,
            risk_score=min(round(prob, 4), 0.04),
            reasons=["Verified legitimate authority domain with valid SSL certificate"]
        )

    if is_lookalike:
        if target_brand and target_token:
            reasons.append(f"Visual typosquatting / lookalike pattern detected targeting '{target_brand}' ('{target_token}')")
        else:
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

    # 5. Determine 3-Level classification adhering strictly to Section 7 & 8 rules
    has_indicators = len(reasons) > 0
    has_harvesting_risk = (
        kw_cnt > 0 or scheme == "http" or brand_in_sub or has_ip or
        susp_tld or has_redirect or has_at or has_homoglyphs or prob >= 0.50
    )

    if is_lookalike:
        if has_harvesting_risk:
            prediction = "Phishing"
            confidence = max(prob, 0.88)
            calibrated_risk = max(prob, 0.85)
            reasons.append(f"High risk: brand lookalike targeting '{target_brand or 'brand'}' with active harvesting/insecure indicators")
        else:
            prediction = "Suspicious"
            confidence = 0.85
            calibrated_risk = max(prob, 0.62)
            reasons.append(f"Suspicious lookalike domain: closely mimics '{target_brand or 'brand'}'; verify authenticity before entering credentials")
    elif prob >= t_phish and has_indicators:
        prediction = "Phishing"
        confidence = prob
        calibrated_risk = prob
        reasons.append(f"Model phishing risk is high ({prob * 100:.1f}%) with supporting malicious patterns")
    elif brand_in_sub or (has_ip and kw_cnt > 0) or has_homoglyphs:
        prediction = "Suspicious" if prob < 0.60 else "Phishing"
        confidence = max(prob, 0.75)
        calibrated_risk = max(prob, 0.65)
    elif prob >= t_phish and not has_indicators:
        prediction = "Suspicious"
        confidence = prob
        calibrated_risk = prob
        reasons.append(f"Elevated statistical risk ({prob * 100:.1f}%) but no confirmed malicious patterns detected; proceed with caution")
    elif prob >= t_safe:
        prediction = "Suspicious"
        confidence = prob
        calibrated_risk = prob
        reasons.append(f"Moderate risk score ({prob * 100:.1f}%) with ambiguous characteristics")
    else:
        prediction = "Safe"
        confidence = 1.0 - prob
        calibrated_risk = prob
        if not reasons:
            reasons.append("No significant suspicious URL patterns or brand impersonation detected")

    return PredictResponse(
        url=url,
        prediction=prediction,
        confidence=round(confidence, 4),
        risk_score=round(calibrated_risk, 4),
        reasons=reasons
    )


# ==========================================
# 2. SPAM VS HAM DETECTOR SCHEMAS & LOGIC
# ==========================================

class SpamPredictRequest(BaseModel):
    text: str = Field(..., description="Message text in English, Arabic, or mixed", min_length=1, max_length=10000)

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "WINNER!! You have been selected to receive a £900 prize reward! Call 09061701461 to claim."
            }
        }
    }


class SpamPredictResponse(BaseModel):
    text: str
    prediction: str = Field(..., description="Classification category: Ham, Suspicious, or Spam")
    confidence: float = Field(..., description="Calibrated confidence score between 0.0 and 1.0")
    risk_score: float = Field(..., description="Spam risk probability between 0.0 and 1.0")
    language: str = Field(..., description="Detected language: English, Arabic, or Mixed")
    reasons: List[str] = Field(default_factory=list, description="Explanations supporting the classification decision")
    security_advice: str = Field(..., description="Protective guidance to protect the user from scams")


# Fallback embedded HTML
INDEX_HTML_CONTENT = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Security AI Suite: Phishing & Spam Detector</title>
  <style>
    :root {
      --bg: #f8fafc;
      --surface: #ffffff;
      --border: #e2e8f0;
      --text-main: #0f172a;
      --text-muted: #64748b;
      --primary: #2563eb;
      --primary-hover: #1d4ed8;
      
      --safe-bg: #f0fdf4;
      --safe-border: #bbf7d0;
      --safe-text: #166534;
      --safe-accent: #15803d;

      --susp-bg: #fffbeb;
      --susp-border: #fde68a;
      --susp-text: #92400e;
      --susp-accent: #b45309;

      --phish-bg: #fff1f2;
      --phish-border: #fecdd3;
      --phish-text: #9f1239;
      --phish-accent: #be123c;
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }

    body {
      background: var(--bg);
      color: var(--text-main);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 2rem 1rem;
    }

    .container {
      width: 100%;
      max-width: 740px;
    }

    .header {
      text-align: center;
      margin-bottom: 1.5rem;
    }

    .header-icon {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 48px;
      height: 48px;
      background: #eff6ff;
      border: 1px solid #bfdbfe;
      border-radius: 12px;
      color: var(--primary);
      margin-bottom: 0.75rem;
    }

    .header h1 {
      font-size: 1.7rem;
      font-weight: 700;
      letter-spacing: -0.02em;
      margin-bottom: 0.35rem;
    }

    .header p {
      font-size: 0.95rem;
      color: var(--text-muted);
    }

    /* Tabs Navigation */
    .nav-tabs {
      display: flex;
      gap: 0.5rem;
      margin-bottom: 1.25rem;
      background: #e2e8f0;
      padding: 0.35rem;
      border-radius: 12px;
      width: 100%;
    }

    .tab-btn {
      flex: 1;
      padding: 0.7rem 1rem;
      border: none;
      background: transparent;
      color: var(--text-muted);
      font-size: 0.88rem;
      font-weight: 600;
      border-radius: 9px;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 0.5rem;
      transition: all 0.2s ease;
    }

    .tab-btn:hover {
      color: var(--text-main);
    }

    .tab-btn.active {
      background: #ffffff;
      color: var(--primary);
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
    }

    .tab-panel {
      display: none;
    }

    .tab-panel.active {
      display: block;
    }

    .card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 1.75rem;
      box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.04);
      margin-bottom: 1.5rem;
    }

    .input-label {
      display: block;
      font-size: 0.875rem;
      font-weight: 600;
      margin-bottom: 0.5rem;
      color: var(--text-main);
    }

    .input-group {
      display: flex;
      gap: 0.5rem;
      margin-bottom: 1rem;
    }

    .url-input {
      flex: 1;
      padding: 0.75rem 1rem;
      border: 1px solid var(--border);
      border-radius: 10px;
      font-size: 0.95rem;
      color: var(--text-main);
      outline: none;
      transition: border-color 0.15s ease, box-shadow 0.15s ease;
      background: #ffffff;
    }

    .url-input:focus, .spam-textarea:focus {
      border-color: var(--primary);
      box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.12);
    }

    .spam-textarea {
      width: 100%;
      padding: 0.85rem 1rem;
      border: 1px solid var(--border);
      border-radius: 10px;
      font-size: 0.95rem;
      color: var(--text-main);
      outline: none;
      resize: vertical;
      min-height: 110px;
      line-height: 1.5;
      margin-bottom: 0.35rem;
      background: #ffffff;
      transition: border-color 0.15s ease, box-shadow 0.15s ease;
    }

    .textarea-footer {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 0.775rem;
      color: var(--text-muted);
      margin-bottom: 1rem;
    }

    .btn-submit {
      padding: 0.75rem 1.4rem;
      background: var(--primary);
      color: #ffffff;
      border: none;
      border-radius: 10px;
      font-size: 0.95rem;
      font-weight: 600;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 0.5rem;
      transition: background 0.15s ease;
      white-space: nowrap;
    }

    .btn-submit:hover {
      background: var(--primary-hover);
    }

    .btn-submit:disabled {
      opacity: 0.6;
      cursor: not-allowed;
    }

    .btn-submit.full-width {
      width: 100%;
      margin-top: 0.5rem;
    }

    .presets {
      display: flex;
      flex-wrap: wrap;
      gap: 0.4rem;
      align-items: center;
      margin-top: 0.75rem;
    }

    .presets-title {
      font-size: 0.775rem;
      color: var(--text-muted);
      margin-right: 0.25rem;
      width: 100%;
      font-weight: 600;
    }

    .preset-pill {
      background: #f1f5f9;
      border: 1px solid var(--border);
      border-radius: 20px;
      padding: 0.25rem 0.65rem;
      font-size: 0.775rem;
      color: var(--text-muted);
      cursor: pointer;
      transition: all 0.15s ease;
    }

    .preset-pill:hover {
      background: #e2e8f0;
      color: var(--text-main);
    }

    /* Result Card */
    .result-box {
      display: none;
      border-radius: 14px;
      padding: 1.5rem;
      margin-top: 1.25rem;
      border: 1px solid transparent;
      animation: fadeIn 0.2s ease-in-out;
    }

    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(4px); }
      to { opacity: 1; transform: translateY(0); }
    }

    .result-box.safe, .result-box.ham {
      background: var(--safe-bg);
      border-color: var(--safe-border);
      color: var(--safe-text);
    }

    .result-box.suspicious {
      background: var(--susp-bg);
      border-color: var(--susp-border);
      color: var(--susp-text);
    }

    .result-box.phishing, .result-box.spam {
      background: var(--phish-bg);
      border-color: var(--phish-border);
      color: var(--phish-text);
    }

    .result-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 1rem;
      flex-wrap: wrap;
      gap: 0.5rem;
    }

    .result-verdict {
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      font-size: 1.25rem;
      font-weight: 700;
    }

    .result-badges-row {
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    .result-badge {
      display: inline-flex;
      align-items: center;
      padding: 0.2rem 0.65rem;
      border-radius: 20px;
      font-size: 0.75rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }

    .lang-badge {
      display: inline-flex;
      align-items: center;
      padding: 0.2rem 0.65rem;
      border-radius: 20px;
      font-size: 0.75rem;
      font-weight: 600;
      background: rgba(0, 0, 0, 0.06);
      color: var(--text-main);
    }

    .result-box.safe .result-badge, .result-box.ham .result-badge {
      background: var(--safe-accent);
      color: #ffffff;
    }

    .result-box.suspicious .result-badge {
      background: var(--susp-accent);
      color: #ffffff;
    }

    .result-box.phishing .result-badge, .result-box.spam .result-badge {
      background: var(--phish-accent);
      color: #ffffff;
    }

    .target-preview {
      font-size: 0.85rem;
      word-break: break-word;
      padding: 0.6rem 0.85rem;
      background: rgba(255, 255, 255, 0.65);
      border-radius: 8px;
      margin-bottom: 1rem;
      font-family: inherit;
      line-height: 1.4;
      border: 1px solid rgba(0, 0, 0, 0.05);
    }

    .metrics-bar-wrap {
      margin-bottom: 1rem;
    }

    .metrics-labels {
      display: flex;
      justify-content: space-between;
      font-size: 0.8rem;
      font-weight: 600;
      margin-bottom: 0.35rem;
    }

    .progress-bar-bg {
      height: 8px;
      background: rgba(0, 0, 0, 0.08);
      border-radius: 4px;
      overflow: hidden;
    }

    .progress-bar-fill {
      height: 100%;
      border-radius: 4px;
      transition: width 0.4s ease;
    }

    .result-box.safe .progress-bar-fill, .result-box.ham .progress-bar-fill {
      background: var(--safe-accent);
    }

    .result-box.suspicious .progress-bar-fill {
      background: var(--susp-accent);
    }

    .result-box.phishing .progress-bar-fill, .result-box.spam .progress-bar-fill {
      background: var(--phish-accent);
    }

    .reasons-wrap {
      margin-top: 0.75rem;
    }

    .reasons-title {
      font-size: 0.8rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.03em;
      margin-bottom: 0.4rem;
      opacity: 0.9;
    }

    .reasons-list {
      list-style: none;
      display: flex;
      flex-direction: column;
      gap: 0.35rem;
      margin-bottom: 1rem;
    }

    .reasons-list li {
      font-size: 0.85rem;
      display: flex;
      align-items: flex-start;
      gap: 0.5rem;
      line-height: 1.35;
    }

    .reasons-list li::before {
      content: "•";
      font-weight: bold;
    }

    .advice-card {
      margin-top: 0.75rem;
      padding: 0.75rem 1rem;
      background: rgba(255, 255, 255, 0.75);
      border-radius: 8px;
      font-size: 0.85rem;
      line-height: 1.45;
      border: 1px solid rgba(0, 0, 0, 0.06);
    }

    .footer {
      text-align: center;
      font-size: 0.825rem;
      color: var(--text-muted);
      margin-top: auto;
    }

    .footer a {
      color: var(--primary);
      text-decoration: none;
    }

    .footer a:hover {
      text-decoration: underline;
    }

    /* Spinner */
    .spinner {
      display: none;
      width: 16px;
      height: 16px;
      border: 2px solid rgba(255, 255, 255, 0.3);
      border-radius: 50%;
      border-top-color: #ffffff;
      animation: spin 0.6s linear infinite;
    }

    @keyframes spin {
      to { transform: rotate(360deg); }
    }
  </style>
</head>
<body>

  <div class="container">
    <header class="header">
      <div class="header-icon">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
          <path d="m9 12 2 2 4-4"/>
        </svg>
      </div>
      <h1>Security AI Protection Suite</h1>
      <p>Detect URL phishing lures & deceptive SMS, chat, or social spam in English and Arabic.</p>
    </header>

    <!-- Top Tabs -->
    <nav class="nav-tabs">
      <button type="button" class="tab-btn active" id="tab-btn-url" onclick="switchTab('url')">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <circle cx="12" cy="12" r="10"/>
          <line x1="2" y1="12" x2="22" y2="12"/>
          <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1 4-10z"/>
        </svg>
        URL Phishing Detector
      </button>
      <button type="button" class="tab-btn" id="tab-btn-spam" onclick="switchTab('spam')">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
        </svg>
        Spam vs Ham Detector (EN & AR)
      </button>
    </nav>

    <!-- TAB 1: URL Phishing Detector -->
    <div id="tab-panel-url" class="tab-panel active">
      <main class="card">
        <form id="analyze-url-form" onsubmit="handleAnalyzeUrl(event)">
          <label for="url-input" class="input-label">Website Address (URL)</label>
          <div class="input-group">
            <input 
              type="text" 
              id="url-input" 
              class="url-input" 
              placeholder="https://example.com" 
              required 
              autocomplete="off"
              spellcheck="false"
            />
            <button type="submit" id="submit-url-btn" class="btn-submit">
              <span class="spinner" id="spinner-url"></span>
              <span id="btn-text-url">Analyze URL</span>
            </button>
          </div>
        </form>

        <div class="presets">
          <span class="presets-title">Quick tests:</span>
          <button type="button" class="preset-pill" onclick="setUrlPreset('https://vf.eg/offers')">vf.eg (Vodafone)</button>
          <button type="button" class="preset-pill" onclick="setUrlPreset('https://www.youtube.com')">youtube.com</button>
          <button type="button" class="preset-pill" onclick="setUrlPreset('https://youtobe.com')">youtobe.com (Lookalike)</button>
          <button type="button" class="preset-pill" onclick="setUrlPreset('https://paypol.com/signin')">paypol.com (Lookalike)</button>
          <button type="button" class="preset-pill" onclick="setUrlPreset('https://univercity.edu')">univercity.edu (Typo)</button>
          <button type="button" class="preset-pill" onclick="setUrlPreset('https://paypal.com.account-verification.xyz/login')">paypal.com.xyz (Spoof)</button>
        </div>

        <!-- URL Result Section -->
        <div id="result-box-url" class="result-box">
          <div class="result-header">
            <div class="result-verdict" id="verdict-url">
              <span id="verdict-icon-url"></span>
              <span id="verdict-text-url">Safe</span>
            </div>
            <span class="result-badge" id="result-badge-url">SAFE</span>
          </div>

          <div class="target-preview" id="result-target-url"></div>

          <div class="metrics-bar-wrap">
            <div class="metrics-labels">
              <span id="conf-label-url">Confidence: 98.0%</span>
              <span id="risk-label-url">Risk Score: 2.0%</span>
            </div>
            <div class="progress-bar-bg">
              <div class="progress-bar-fill" id="progress-fill-url" style="width: 2%;"></div>
            </div>
          </div>

          <div class="reasons-wrap">
            <div class="reasons-title">Findings & Security Indicators</div>
            <ul class="reasons-list" id="reasons-list-url"></ul>
          </div>
        </div>
      </main>
    </div>

    <!-- TAB 2: Spam vs Ham Message Detector -->
    <div id="tab-panel-spam" class="tab-panel">
      <main class="card">
        <form id="analyze-spam-form" onsubmit="handleAnalyzeSpam(event)">
          <label for="spam-text-input" class="input-label">Message Content / نص الرسالة (English or Arabic)</label>
          <textarea 
            id="spam-text-input" 
            class="spam-textarea" 
            placeholder="Type or paste any SMS, WhatsApp message, email, or social post to inspect... / اكتب أو الصق الرسالة هنا للفحص..."
            required
            maxlength="10000"
            oninput="updateCharCount()"
          ></textarea>
          <div class="textarea-footer">
            <span>Supports English SMS & Arabic Tweets/Texts</span>
            <span id="char-counter">0 / 10000</span>
          </div>

          <button type="submit" id="submit-spam-btn" class="btn-submit full-width">
            <span class="spinner" id="spinner-spam"></span>
            <span id="btn-text-spam">Analyze Message / فحص الرسالة</span>
          </button>
        </form>

        <div class="presets">
          <span class="presets-title">Quick test samples:</span>
          <button type="button" class="preset-pill" onclick="setSpamPreset('WINNER!! You have been selected to receive a £900 prize reward! Call 09061701461 to claim code KL341. Valid 12h only.')">🇬🇧 Prize Scam</button>
          <button type="button" class="preset-pill" onclick="setSpamPreset('URGENT: Your bank card has been suspended due to unusual activity. Verify your identity immediately: http://bit.ly/bank-verify')">🇬🇧 Urgent Bank Phish</button>
          <button type="button" class="preset-pill" onclick="setSpamPreset('Hey, are we still meeting for lunch tomorrow at 2pm near the library?')">🇬🇧 Normal Chat</button>
          <button type="button" class="preset-pill" onclick="setSpamPreset('مبروك ربحت معنا 50000 ريال استلم جائزتك الان فوراً عبر الرابط التالي')">🇸🇦 احتيال جائزة مالية</button>
          <button type="button" class="preset-pill" onclick="setSpamPreset('عاجل تم ايقاف حسابك البنكي يرجى تحديث بياناتك فورا لتجنب تجميد الرصيد عبر واتساب')">🇸🇦 تنبيه بنكي مزيف</button>
          <button type="button" class="preset-pill" onclick="setSpamPreset('السلام عليكم ورحمة الله، كيف حالك يا أخي؟ نلتقي غداً في الجامعة إن شاء الله')">🇸🇦 رسالة عادية</button>
        </div>

        <!-- Spam Result Section -->
        <div id="result-box-spam" class="result-box">
          <div class="result-header">
            <div class="result-verdict" id="verdict-spam">
              <span id="verdict-icon-spam"></span>
              <span id="verdict-text-spam">Ham</span>
            </div>
            <div class="result-badges-row">
              <span class="lang-badge" id="lang-badge-spam">English</span>
              <span class="result-badge" id="result-badge-spam">HAM</span>
            </div>
          </div>

          <div class="target-preview" id="result-target-spam"></div>

          <div class="metrics-bar-wrap">
            <div class="metrics-labels">
              <span id="conf-label-spam">Confidence: 99.0%</span>
              <span id="risk-label-spam">Spam Probability: 1.0%</span>
            </div>
            <div class="progress-bar-bg">
              <div class="progress-bar-fill" id="progress-fill-spam" style="width: 1%;"></div>
            </div>
          </div>

          <div class="reasons-wrap">
            <div class="reasons-title">Threat Indicators & Signals / المؤشرات</div>
            <ul class="reasons-list" id="reasons-list-spam"></ul>
          </div>

          <div class="advice-card" id="advice-card-spam"></div>
        </div>
      </main>
    </div>

    <footer class="footer">
      <p>
        API: <a href="/docs" target="_blank">Swagger Docs</a> &bull; 
        <a href="/health" target="_blank">Health Check</a> &bull; 
        Powered by Random Forest (URL) & Logistic Regression (Spam/Ham)
      </p>
    </footer>
  </div>

  <script>
    function switchTab(tab) {
      document.getElementById('tab-btn-url').classList.toggle('active', tab === 'url');
      document.getElementById('tab-btn-spam').classList.toggle('active', tab === 'spam');
      document.getElementById('tab-panel-url').classList.toggle('active', tab === 'url');
      document.getElementById('tab-panel-spam').classList.toggle('active', tab === 'spam');
    }

    function updateCharCount() {
      const text = document.getElementById('spam-text-input').value;
      document.getElementById('char-counter').textContent = text.length + ' / 10000';
    }

    // --- URL Detector Logic ---
    function setUrlPreset(url) {
      document.getElementById('url-input').value = url;
      analyzeUrl(url);
    }

    function handleAnalyzeUrl(e) {
      e.preventDefault();
      const url = document.getElementById('url-input').value.trim();
      if (url) {
        analyzeUrl(url);
      }
    }

    async function analyzeUrl(url) {
      const btn = document.getElementById('submit-url-btn');
      const spinner = document.getElementById('spinner-url');
      const btnText = document.getElementById('btn-text-url');

      btn.disabled = true;
      spinner.style.display = 'inline-block';
      btnText.textContent = 'Analyzing...';

      try {
        const response = await fetch('/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ url: url })
        });

        if (!response.ok) {
          const errData = await response.json().catch(() => ({}));
          throw new Error(errData.detail || 'Analysis request failed');
        }

        const data = await response.json();
        renderUrlResult(data);
      } catch (err) {
        alert('Error: ' + err.message);
      } finally {
        btn.disabled = false;
        spinner.style.display = 'none';
        btnText.textContent = 'Analyze URL';
      }
    }

    function renderUrlResult(data) {
      const box = document.getElementById('result-box-url');
      const verdictText = document.getElementById('verdict-text-url');
      const verdictIcon = document.getElementById('verdict-icon-url');
      const badge = document.getElementById('result-badge-url');
      const targetDisplay = document.getElementById('result-target-url');
      const confLabel = document.getElementById('conf-label-url');
      const riskLabel = document.getElementById('risk-label-url');
      const progressFill = document.getElementById('progress-fill-url');
      const reasonsList = document.getElementById('reasons-list-url');

      const pred = data.prediction.toLowerCase();
      box.className = 'result-box ' + pred;
      box.style.display = 'block';

      verdictText.textContent = data.prediction;
      badge.textContent = data.prediction.toUpperCase();
      targetDisplay.textContent = data.url;

      const confPercent = Math.round(data.confidence * 1000) / 10;
      const riskPercent = Math.round(data.risk_score * 1000) / 10;

      confLabel.textContent = `Confidence: ${confPercent}%`;
      riskLabel.textContent = `Risk Score: ${riskPercent}%`;

      if (pred === 'safe') {
        progressFill.style.width = Math.max(5, riskPercent) + '%';
        verdictIcon.innerHTML = `
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
            <path d="m9 12 2 2 4-4"/>
          </svg>
        `;
      } else if (pred === 'suspicious') {
        progressFill.style.width = Math.max(15, riskPercent) + '%';
        verdictIcon.innerHTML = `
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
            <line x1="12" y1="9" x2="12" y2="13"/>
            <line x1="12" y1="17" x2="12.01" y2="17"/>
          </svg>
        `;
      } else {
        progressFill.style.width = Math.max(20, riskPercent) + '%';
        verdictIcon.innerHTML = `
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="12" cy="12" r="10"/>
            <line x1="15" y1="9" x2="9" y2="15"/>
            <line x1="9" y1="9" x2="15" y2="15"/>
          </svg>
        `;
      }

      reasonsList.innerHTML = '';
      if (data.reasons && data.reasons.length > 0) {
        data.reasons.forEach(r => {
          const li = document.createElement('li');
          li.textContent = r;
          reasonsList.appendChild(li);
        });
      } else {
        const li = document.createElement('li');
        li.textContent = 'No abnormal characteristics detected.';
        reasonsList.appendChild(li);
      }
    }

    // --- Spam vs Ham Detector Logic ---
    function setSpamPreset(text) {
      document.getElementById('spam-text-input').value = text;
      updateCharCount();
      analyzeSpam(text);
    }

    function handleAnalyzeSpam(e) {
      e.preventDefault();
      const text = document.getElementById('spam-text-input').value.trim();
      if (text) {
        analyzeSpam(text);
      }
    }

    async function analyzeSpam(text) {
      const btn = document.getElementById('submit-spam-btn');
      const spinner = document.getElementById('spinner-spam');
      const btnText = document.getElementById('btn-text-spam');

      btn.disabled = true;
      spinner.style.display = 'inline-block';
      btnText.textContent = 'Inspecting message...';

      try {
        const response = await fetch('/predict-spam', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: text })
        });

        if (!response.ok) {
          const errData = await response.json().catch(() => ({}));
          throw new Error(errData.detail || 'Spam analysis request failed');
        }

        const data = await response.json();
        renderSpamResult(data);
      } catch (err) {
        alert('Error: ' + err.message);
      } finally {
        btn.disabled = false;
        spinner.style.display = 'none';
        btnText.textContent = 'Analyze Message / فحص الرسالة';
      }
    }

    function renderSpamResult(data) {
      const box = document.getElementById('result-box-spam');
      const verdictText = document.getElementById('verdict-text-spam');
      const verdictIcon = document.getElementById('verdict-icon-spam');
      const badge = document.getElementById('result-badge-spam');
      const langBadge = document.getElementById('lang-badge-spam');
      const targetDisplay = document.getElementById('result-target-spam');
      const confLabel = document.getElementById('conf-label-spam');
      const riskLabel = document.getElementById('risk-label-spam');
      const progressFill = document.getElementById('progress-fill-spam');
      const reasonsList = document.getElementById('reasons-list-spam');
      const adviceCard = document.getElementById('advice-card-spam');

      const pred = data.prediction.toLowerCase();
      box.className = 'result-box ' + pred;
      box.style.display = 'block';

      verdictText.textContent = data.prediction === 'Ham' ? 'Safe (Ham)' : data.prediction;
      badge.textContent = data.prediction.toUpperCase();
      langBadge.textContent = data.language || 'Detected';
      targetDisplay.textContent = data.text;

      const confPercent = Math.round(data.confidence * 1000) / 10;
      const riskPercent = Math.round(data.risk_score * 1000) / 10;

      confLabel.textContent = `Confidence: ${confPercent}%`;
      riskLabel.textContent = `Spam Probability: ${riskPercent}%`;

      if (pred === 'ham') {
        progressFill.style.width = Math.max(5, riskPercent) + '%';
        verdictIcon.innerHTML = `
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
            <path d="m9 12 2 2 4-4"/>
          </svg>
        `;
      } else if (pred === 'suspicious') {
        progressFill.style.width = Math.max(15, riskPercent) + '%';
        verdictIcon.innerHTML = `
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
            <line x1="12" y1="9" x2="12" y2="13"/>
            <line x1="12" y1="17" x2="12.01" y2="17"/>
          </svg>
        `;
      } else {
        progressFill.style.width = Math.max(20, riskPercent) + '%';
        verdictIcon.innerHTML = `
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="12" cy="12" r="10"/>
            <line x1="15" y1="9" x2="9" y2="15"/>
            <line x1="9" y1="9" x2="15" y2="15"/>
          </svg>
        `;
      }

      reasonsList.innerHTML = '';
      if (data.reasons && data.reasons.length > 0) {
        data.reasons.forEach(r => {
          const li = document.createElement('li');
          li.textContent = r;
          reasonsList.appendChild(li);
        });
      } else {
        const li = document.createElement('li');
        li.textContent = 'No suspicious triggers detected.';
        reasonsList.appendChild(li);
      }

      adviceCard.textContent = data.security_advice || '';
    }
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse, summary="Web Interface")
def get_root():
    """Serves the modern, clean tabbed web interface."""
    # Attempt to read freshest index.html from disk, falling back to embedded string
    content = INDEX_HTML_CONTENT
    if os.path.exists(INDEX_HTML_PATH):
        try:
            with open(INDEX_HTML_PATH, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception:
            pass

    return HTMLResponse(
        content=content,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )


@app.get("/api", summary="API Information")
def get_api_info():
    """Information endpoint providing API metadata and links."""
    return {
        "name": "Security AI Protection Suite API",
        "version": "1.1.0",
        "status": "online",
        "description": "High-performance, lightweight URL Phishing & Multilingual Spam/Ham Detection service.",
        "models": {
            "url_phishing_model": {
                "loaded": MODEL_STATE["loaded"],
                "type": MODEL_STATE["model_type"],
                "features": len(MODEL_STATE["feature_names"] or [])
            },
            "spam_detector_model": {
                "loaded": SPAM_MODEL_STATE["loaded"],
                "supported_languages": ["English", "Arabic", "Mixed"]
            }
        },
        "endpoints": {
            "web_ui": "/",
            "api_info": "/api",
            "health": "/health",
            "predict_url": "/predict",
            "predict_spam": "/predict-spam",
            "docs": "/docs"
        }
    }


@app.get("/health", summary="Health Check")
def get_health():
    """Health check endpoint indicating service and models readiness."""
    url_loaded = MODEL_STATE["loaded"]
    spam_loaded = SPAM_MODEL_STATE["loaded"]

    overall_status = "healthy" if (url_loaded and spam_loaded) else ("degraded" if (url_loaded or spam_loaded) else "offline")

    return {
        "status": overall_status,
        "url_model_loaded": url_loaded,
        "spam_model_loaded": spam_loaded,
        "version": "1.1.0",
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
            detail=f"An error occurred during URL prediction: {str(e)}"
        )


@app.post("/predict-spam", response_model=SpamPredictResponse, summary="Classify Message (English & Arabic)")
def predict_spam(payload: SpamPredictRequest):
    """
    Accepts a message text (SMS, WhatsApp, email, tweet) in English, Arabic, or mixed,
    and classifies it into Ham, Suspicious, or Spam with explainable reasons and protection advice.
    """
    clean_text = payload.text.strip()
    if not clean_text:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Message text must not be empty or whitespace only."
        )

    try:
        res_dict = classify_message(clean_text)
        return SpamPredictResponse(**res_dict)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during spam prediction: {str(e)}"
        )


@app.post("/predict_spam", response_model=SpamPredictResponse, include_in_schema=False)
def predict_spam_alias(payload: SpamPredictRequest):
    """Alias for /predict-spam endpoint."""
    return predict_spam(payload)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
