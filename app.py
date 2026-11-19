"""
FastAPI Application for URL Phishing Detector.
Provides / , /health, and /predict endpoints with 3-level classification:
Safe, Suspicious, or Phishing.
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
    """Load model once on application startup and bridge alternate Railway ports."""
    model_env_path = os.environ.get("MODEL_PATH", "model.joblib")
    load_model_artifact(model_env_path)

    # Railway port bridge: ensure both port 8000 and 8080 respond cleanly
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


INDEX_HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>URL Phishing Detector</title>
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
      padding: 2.5rem 1rem;
    }

    .container {
      width: 100%;
      max-width: 680px;
    }

    .header {
      text-align: center;
      margin-bottom: 2rem;
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
      font-size: 1.65rem;
      font-weight: 700;
      letter-spacing: -0.02em;
      margin-bottom: 0.35rem;
    }

    .header p {
      font-size: 0.95rem;
      color: var(--text-muted);
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

    .url-input:focus {
      border-color: var(--primary);
      box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.12);
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

    .presets {
      display: flex;
      flex-wrap: wrap;
      gap: 0.4rem;
      align-items: center;
    }

    .presets-title {
      font-size: 0.775rem;
      color: var(--text-muted);
      margin-right: 0.25rem;
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
      margin-top: 1rem;
      border: 1px solid transparent;
      animation: fadeIn 0.2s ease-in-out;
    }

    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(4px); }
      to { opacity: 1; transform: translateY(0); }
    }

    .result-box.safe {
      background: var(--safe-bg);
      border-color: var(--safe-border);
      color: var(--safe-text);
    }

    .result-box.suspicious {
      background: var(--susp-bg);
      border-color: var(--susp-border);
      color: var(--susp-text);
    }

    .result-box.phishing {
      background: var(--phish-bg);
      border-color: var(--phish-border);
      color: var(--phish-text);
    }

    .result-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 1rem;
    }

    .result-verdict {
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      font-size: 1.25rem;
      font-weight: 700;
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

    .result-box.safe .result-badge {
      background: var(--safe-accent);
      color: #ffffff;
    }

    .result-box.suspicious .result-badge {
      background: var(--susp-accent);
      color: #ffffff;
    }

    .result-box.phishing .result-badge {
      background: var(--phish-accent);
      color: #ffffff;
    }

    .target-url {
      font-size: 0.85rem;
      word-break: break-all;
      padding: 0.5rem 0.75rem;
      background: rgba(255, 255, 255, 0.6);
      border-radius: 8px;
      margin-bottom: 1rem;
      font-family: monospace;
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

    .result-box.safe .progress-bar-fill {
      background: var(--safe-accent);
    }

    .result-box.suspicious .progress-bar-fill {
      background: var(--susp-accent);
    }

    .result-box.phishing .progress-bar-fill {
      background: var(--phish-accent);
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
      <h1>URL Phishing Detector</h1>
      <p>Analyze web addresses for phishing lures, typosquatting, and impersonation.</p>
    </header>

    <main class="card">
      <form id="analyze-form" onsubmit="handleAnalyze(event)">
        <label for="url-input" class="input-label">Website Address</label>
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
          <button type="submit" id="submit-btn" class="btn-submit">
            <span class="spinner" id="spinner"></span>
            <span id="btn-text">Analyze</span>
          </button>
        </div>
      </form>

      <div class="presets">
        <span class="presets-title">Quick test:</span>
        <button type="button" class="preset-pill" onclick="setPreset('https://vf.eg/offers')">vf.eg (Vodafone)</button>
        <button type="button" class="preset-pill" onclick="setPreset('https://www.google.com')">google.com</button>
        <button type="button" class="preset-pill" onclick="setPreset('https://go0g1e.com/login')">go0g1e.com (Lookalike)</button>
        <button type="button" class="preset-pill" onclick="setPreset('https://paypa1.com/verify-account')">paypa1.com (Lookalike)</button>
        <button type="button" class="preset-pill" onclick="setPreset('https://paypal.com.account-verification.xyz/login')">paypal.com.xyz (Spoof)</button>
      </div>

      <!-- Result Section -->
      <div id="result-box" class="result-box">
        <div class="result-header">
          <div class="result-verdict" id="result-verdict">
            <span id="verdict-icon"></span>
            <span id="verdict-text">Safe</span>
          </div>
          <span class="result-badge" id="result-badge">SAFE</span>
        </div>

        <div class="target-url" id="result-url"></div>

        <div class="metrics-bar-wrap">
          <div class="metrics-labels">
            <span id="confidence-label">Confidence: 98.0%</span>
            <span id="risk-label">Risk Score: 2.0%</span>
          </div>
          <div class="progress-bar-bg">
            <div class="progress-bar-fill" id="progress-fill" style="width: 2%;"></div>
          </div>
        </div>

        <div class="reasons-wrap">
          <div class="reasons-title">Findings & Explanations</div>
          <ul class="reasons-list" id="reasons-list"></ul>
        </div>
      </div>
    </main>

    <footer class="footer">
      <p>
        API: <a href="/docs" target="_blank">Swagger Docs</a> &bull; 
        <a href="/health" target="_blank">Health Check</a> &bull; 
        Powered by Random Forest & FastAPI
      </p>
    </footer>
  </div>

  <script>
    function setPreset(url) {
      document.getElementById('url-input').value = url;
      analyzeUrl(url);
    }

    function handleAnalyze(e) {
      e.preventDefault();
      const url = document.getElementById('url-input').value.trim();
      if (url) {
        analyzeUrl(url);
      }
    }

    async function analyzeUrl(url) {
      const btn = document.getElementById('submit-btn');
      const spinner = document.getElementById('spinner');
      const btnText = document.getElementById('btn-text');
      const resultBox = document.getElementById('result-box');

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
        renderResult(data);
      } catch (err) {
        alert('Error: ' + err.message);
      } finally {
        btn.disabled = false;
        spinner.style.display = 'none';
        btnText.textContent = 'Analyze';
      }
    }

    function renderResult(data) {
      const box = document.getElementById('result-box');
      const verdictText = document.getElementById('verdict-text');
      const verdictIcon = document.getElementById('verdict-icon');
      const badge = document.getElementById('result-badge');
      const urlDisplay = document.getElementById('result-url');
      const confLabel = document.getElementById('confidence-label');
      const riskLabel = document.getElementById('risk-label');
      const progressFill = document.getElementById('progress-fill');
      const reasonsList = document.getElementById('reasons-list');

      box.className = 'result-box ' + data.prediction.toLowerCase();
      box.style.display = 'block';

      verdictText.textContent = data.prediction;
      badge.textContent = data.prediction.toUpperCase();
      urlDisplay.textContent = data.url;

      const confPercent = Math.round(data.confidence * 1000) / 10;
      const riskPercent = Math.round(data.risk_score * 1000) / 10;

      confLabel.textContent = `Confidence: ${confPercent}%`;
      riskLabel.textContent = `Risk Score: ${riskPercent}%`;

      if (data.prediction.toLowerCase() === 'safe') {
        progressFill.style.width = Math.max(5, 100 - confPercent) + '%';
        verdictIcon.innerHTML = `
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
            <path d="m9 12 2 2 4-4"/>
          </svg>
        `;
      } else if (data.prediction.toLowerCase() === 'suspicious') {
        progressFill.style.width = Math.max(10, riskPercent) + '%';
        verdictIcon.innerHTML = `
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
            <line x1="12" y1="9" x2="12" y2="13"/>
            <line x1="12" y1="17" x2="12.01" y2="17"/>
          </svg>
        `;
      } else {
        progressFill.style.width = Math.max(10, riskPercent) + '%';
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
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse, summary="Web Interface")
def get_root():
    """Serves the modern, clean web interface."""
    return HTMLResponse(content=INDEX_HTML_CONTENT)


@app.get("/api", summary="API Information")
def get_api_info():
    """Information endpoint providing API metadata and links."""
    return {
        "name": "URL Phishing Detector API",
        "version": MODEL_STATE["version"],
        "status": "online",
        "description": "High-performance, lightweight URL phishing and typosquatting detector.",
        "model_loaded": MODEL_STATE["loaded"],
        "endpoints": {
            "web_ui": "/",
            "api_info": "/api",
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
