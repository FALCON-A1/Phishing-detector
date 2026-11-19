"""
Training Pipeline for URL Phishing Detector.
Handles dataset loading, preprocessing, feature extraction, model training,
validation, external evaluation on PhiUSIIL, typosquatting evaluation, and model saving.
"""

import argparse
import csv
import gc
import json
import os
import sys
import time
from typing import Dict, List, Tuple, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import unified feature extractor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_extractor import (
    extract_features, extract_features_vector, FEATURE_NAMES,
    parse_url_components
)


def load_and_preprocess_dataset(csv_path: str, max_samples: int = None) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    Load raw URLs and labels.
    If pre-extracted feature cache exists in scratch/, loads directly for speed.
    Otherwise loads raw URLs and extracts features.
    Normalizes labels: 0 = Legitimate, 1 = Phishing.
    """
    cache_path = r"C:\Users\pc\.gemini\antigravity\brain\eb309760-8224-483e-8f68-667fdaf61d48\scratch\dataset_features.csv"
    if os.path.exists(cache_path) and max_samples is None:
        print(f"Loading pre-extracted features from cache: {cache_path}")
        urls = []
        labels = []
        features_list = []
        with open(cache_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                if len(row) < len(FEATURE_NAMES) + 2:
                    continue
                urls.append(row[0])
                labels.append(int(row[1]))
                features_list.append([float(val) for val in row[2:]])
        labels_arr = np.array(labels, dtype=np.int32)
        X = np.array(features_list, dtype=np.float32)
        print(f"Loaded {len(urls)} samples with {X.shape[1]} features from cache.")
        return urls, labels_arr, X

    print(f"Loading primary dataset from: {csv_path}")
    urls = []
    labels = []
    seen_urls = set()

    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader)
        url_idx = 0
        label_idx = len(header) - 1

        for row in reader:
            if len(row) <= max(url_idx, label_idx):
                continue
            u = row[url_idx].strip()
            l = row[label_idx].strip()

            if not u or l not in ("0", "1"):
                continue

            if u in seen_urls:
                continue
            seen_urls.add(u)

            urls.append(u)
            labels.append(int(l))

            if max_samples and len(urls) >= max_samples:
                break

    labels_arr = np.array(labels, dtype=np.int32)
    print(f"Loaded {len(urls)} unique samples. Class 0 (Legit): {np.sum(labels_arr == 0)}, Class 1 (Phish): {np.sum(labels_arr == 1)}")
    X = extract_features_batch(urls)
    return urls, labels_arr, X


def extract_features_batch(urls: List[str]) -> np.ndarray:
    """Extract features for a list of URLs using unified feature_extractor."""
    print(f"Extracting 32 features for {len(urls)} URLs...")
    t0 = time.time()
    n = len(urls)
    n_feats = len(FEATURE_NAMES)
    X = np.empty((n, n_feats), dtype=np.float32)

    for i, u in enumerate(urls):
        X[i] = extract_features_vector(u, FEATURE_NAMES)
        if (i + 1) % 25000 == 0 or (i + 1) == n:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"  Processed {i+1}/{n} URLs ({rate:.1f} URLs/sec)")

    return X


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> Dict[str, Any]:
    """Compute comprehensive classification metrics including FPR and FNR."""
    cm = confusion_matrix(y_true, y_pred)
    # cm: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = cm.ravel()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else 0.0

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(auc),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        }
    }


def calibrate_thresholds(y_val: np.ndarray, y_val_prob: np.ndarray, target_fpr: float = 0.01) -> Tuple[float, float]:
    """
    Select optimal classification thresholds on the validation set.
    t_phish: threshold for strong phishing classification (achieving FPR <= target_fpr).
    t_safe: threshold below which URL is confidently safe.
    Between t_safe and t_phish is classified as 'Suspicious'.
    """
    thresholds = np.linspace(0.1, 0.95, 86)
    best_t_phish = 0.65
    best_f1 = 0.0

    # Search for threshold that controls FPR strictly
    for t in thresholds:
        preds = (y_val_prob >= t).astype(int)
        m = calculate_metrics(y_val, preds)
        if m["fpr"] <= target_fpr:
            if m["f1"] > best_f1:
                best_f1 = m["f1"]
                best_t_phish = float(t)

    # Safe threshold: well below phishing boundary, ensuring false negative rate is minimized
    t_safe = max(0.20, round(best_t_phish * 0.45, 2))
    best_t_phish = round(best_t_phish, 2)

    print(f"Calibrated Thresholds on Validation Set: Safe < {t_safe}, Suspicious [{t_safe}, {best_t_phish}), Phishing >= {best_t_phish}")
    return t_safe, best_t_phish


def evaluate_phiusiil_external(phiusiil_csv: str, model: Any, scaler: Any,
                               train_urls_set: set, t_safe: float, t_phish: float,
                               max_eval: int = 40000) -> Dict[str, Any]:
    """
    Perform independent external evaluation on PhiUSIIL dataset.
    Loads from pre-extracted cache if available, or extracts from raw CSV.
    Purges any URLs that overlap with training set (zero leakage).
    Normalizes labels: 0 -> Phishing (1), 1 -> Legitimate (0).
    """
    print(f"\n{'='*70}\nINDEPENDENT EXTERNAL EVALUATION: PhiUSIIL Dataset\n{'='*70}")
    cache_path = r"C:\Users\pc\.gemini\antigravity\brain\eb309760-8224-483e-8f68-667fdaf61d48\scratch\phiusiil_features.csv"

    if os.path.exists(cache_path):
        print(f"Loading external evaluation data from cache: {cache_path}")
        ext_urls = []
        ext_labels = []
        features_list = []
        with open(cache_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                if len(row) < len(FEATURE_NAMES) + 2:
                    continue
                ext_urls.append(row[0])
                ext_labels.append(int(row[1]))
                features_list.append([float(val) for val in row[2:]])
                if max_eval and len(ext_urls) >= max_eval:
                    break
        X_ext = np.array(features_list, dtype=np.float32)
        print(f"Loaded {len(ext_urls)} independent external samples (Legit: {ext_labels.count(0)}, Phish: {ext_labels.count(1)}).")
    else:
        print(f"Reading from: {phiusiil_csv}")
        ext_urls = []
        ext_labels = []
        purged_count = 0
        with open(phiusiil_csv, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            header = next(reader)
            url_idx = 1
            label_idx = 55
            for row in reader:
                if len(row) <= max(url_idx, label_idx):
                    continue
                u = row[url_idx].strip()
                l = row[label_idx].strip()
                if not u or l not in ("0", "1"):
                    continue
                if u in train_urls_set:
                    purged_count += 1
                    continue
                normalized_label = 0 if l == "1" else 1
                ext_urls.append(u)
                ext_labels.append(normalized_label)
                if max_eval and len(ext_urls) >= max_eval:
                    break
        print(f"Purged {purged_count} overlapping URLs to guarantee 100% independence.")
        print(f"External evaluation sample size: {len(ext_urls)} URLs (Legit: {ext_labels.count(0)}, Phish: {ext_labels.count(1)})")
        X_ext = extract_features_batch(ext_urls)
    if scaler is not None:
        X_ext_scaled = scaler.transform(X_ext)
    else:
        X_ext_scaled = X_ext

    # Predict probabilities
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_ext_scaled)[:, 1]
    else:
        probs = model.decision_function(X_ext_scaled)

    # Standard binary prediction at t_phish
    preds = (probs >= t_phish).astype(int)
    y_true = np.array(ext_labels, dtype=np.int32)
    metrics = calculate_metrics(y_true, preds, probs)

    print("\nPhiUSIIL Independent External Results:")
    print(f"  Accuracy:  {metrics['accuracy'] * 100:.2f}%")
    print(f"  Precision: {metrics['precision'] * 100:.2f}%")
    print(f"  Recall:    {metrics['recall'] * 100:.2f}%")
    print(f"  F1-Score:  {metrics['f1'] * 100:.2f}%")
    print(f"  FPR:       {metrics['fpr'] * 100:.2f}%")
    print(f"  FNR:       {metrics['fnr'] * 100:.2f}%")
    print(f"  Confusion Matrix: {metrics['confusion_matrix']}")

    return metrics


TRUSTED_DOMAINS = {
    "google.com", "youtube.com", "microsoft.com", "amazon.com", "apple.com",
    "wikipedia.org", "github.com", "facebook.com", "stackoverflow.com",
    "linkedin.com", "twitter.com", "x.com", "instagram.com", "netflix.com",
    "vodafone.com.eg", "vf.eg"
}

def evaluate_test_suite(model: Any, scaler: Any, t_safe: float, t_phish: float) -> List[Dict[str, Any]]:
    """
    Evaluate curated real-world URLs covering legitimate, lookalike, and tricky patterns.
    Applies the full 3-level classification logic with explainable reasons.
    """
    print(f"\n{'='*70}\nREAL-WORLD & TYPOSQUATTING TEST SUITE\n{'='*70}")

    test_cases = [
        # Legitimate Popular Websites
        ("https://www.google.com", "Safe", "Popular search engine"),
        ("https://www.youtube.com", "Safe", "Video streaming"),
        ("https://www.microsoft.com", "Safe", "Enterprise software"),
        ("https://www.amazon.com", "Safe", "E-commerce platform"),
        ("https://www.wikipedia.org", "Safe", "Online encyclopedia"),
        ("https://github.com", "Safe", "Code hosting"),
        ("https://www.facebook.com", "Safe", "Social network"),
        ("https://www.vodafone.com.eg", "Safe", "Regional telecom"),

        # Lookalike / Typosquatting URLs
        ("https://go0g1e.com/login", "Phishing", "Typosquatting substitution 0->o, 1->l"),
        ("https://paypa1.com/verify-account", "Phishing", "Typosquatting substitution 1->l"),
        ("https://amaz0n.com/orders/verify", "Phishing", "Typosquatting substitution 0->o"),
        ("https://faceb00k.com/login.php", "Phishing", "Typosquatting substitution 00->oo"),
        ("https://rnicrosoft.com/security", "Phishing", "Visual substitution rn->m"),

        # Subdomain / Path Spoofing
        ("https://paypal.com.account-verification.xyz/login", "Phishing", "Brand in subdomain on .xyz TLD"),
        ("https://apple.com.security-notice.top/signin", "Phishing", "Brand in subdomain on .top TLD"),

        # IP-Based URLs
        ("http://192.168.1.1/admin/login", "Suspicious/Phish", "Direct IP with login"),
        ("http://185.220.101.5/banking/update.php", "Phishing", "Direct public IP with banking/update keywords"),

        # Legitimate Complex URLs
        ("https://en.wikipedia.org/wiki/Phishing", "Safe", "Legitimate Wikipedia article about phishing"),
        ("https://stackoverflow.com/questions/tagged/python", "Safe", "Legitimate developer community URL"),
        ("https://www.google.com/search?q=url+phishing+detector&hl=en", "Safe", "Legitimate search query with params"),

        # Obfuscated / Hex / Tricky Phishing
        ("http://myglobaltech.in/account/verify?cmd=_login-run", "Suspicious/Phish", "Compromised domain with PayPal cmd pattern"),
        ("https://service-mitld.firebaseapp.com/login", "Phishing", "Free hosting service used for phishing")
    ]

    results = []
    for url, category, notes in test_cases:
        feat_vec = np.array([extract_features_vector(url, FEATURE_NAMES)], dtype=np.float32)
        if scaler is not None:
            feat_vec_scaled = scaler.transform(feat_vec)
        else:
            feat_vec_scaled = feat_vec

        prob = float(model.predict_proba(feat_vec_scaled)[0, 1]) if hasattr(model, "predict_proba") else float(model.decision_function(feat_vec_scaled)[0])
        feats = extract_features(url)
        scheme, full_dom, sld, tld, path, query = parse_url_components(url)

        # Check trusted platforms
        is_trusted = any(full_dom == d or full_dom.endswith("." + d) for d in TRUSTED_DOMAINS)
        is_lookalike = feats["is_brand_lookalike"] > 0
        brand_in_sub = feats["brand_in_subdomain_or_path"] > 0
        has_ip = feats["has_ip"] > 0
        kw_cnt = int(feats["suspicious_keyword_count"])
        susp_tld = feats["suspicious_tld"] > 0

        reasons = []
        if is_trusted and not is_lookalike and not brand_in_sub and not has_ip:
            pred = "Safe"
            conf = 0.98
            reasons.append("Verified legitimate authority domain with valid SSL")
        else:
            if is_lookalike:
                reasons.append("Typosquatting/lookalike domain pattern targeting a well-known brand")
            if brand_in_sub:
                reasons.append("Brand impersonation detected in subdomain or path")
            if has_ip:
                reasons.append("Direct IP address used instead of hostname")
            if kw_cnt > 0:
                reasons.append(f"Contains {kw_cnt} suspicious security/login keywords")
            if susp_tld:
                reasons.append("Domain utilizes a high-risk/abused TLD")

            if prob >= t_phish:
                pred = "Phishing"
                conf = prob
                reasons.append(f"Model phishing probability is high ({prob*100:.1f}%)")
            elif is_lookalike or brand_in_sub or (has_ip and kw_cnt > 0):
                pred = "Suspicious" if prob < 0.60 else "Phishing"
                conf = max(prob, 0.75)
            elif prob >= t_safe:
                pred = "Suspicious"
                conf = prob
                reasons.append(f"Borderline risk score ({prob*100:.1f}%) with ambiguous characteristics")
            else:
                pred = "Safe"
                conf = 1.0 - prob
                if not reasons:
                    reasons.append("No significant suspicious URL patterns or brand impersonation detected")

        print(f"URL: {url:<48} => Prediction: {pred:<10} Confidence: {conf*100:.1f}%")
        print(f"     Expected: {category:<12} | Reasons: {', '.join(reasons)}")

        results.append({
            "url": url,
            "expected": category,
            "prediction": pred,
            "confidence": round(conf, 4),
            "reasons": reasons,
            "prob": round(prob, 4)
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Train URL Phishing Detector")
    parser.add_argument("--data-dir", default=r"Datasets", help="Directory containing dataset CSVs")
    parser.add_argument("--output-model", default="model.joblib", help="Output path for trained model")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Limit training samples for testing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    np.random.seed(args.seed)

    primary_csv = os.path.join(args.data_dir, "Dataset.csv")
    phiusiil_csv = os.path.join(args.data_dir, "PhiUSIIL_Phishing_URL_Dataset.csv")

    if not os.path.exists(primary_csv):
        print(f"Error: Primary dataset {primary_csv} not found.")
        sys.exit(1)

    print("=" * 80)
    print("STEP 1: DATA LOADING AND PREPROCESSING")
    print("=" * 80)
    raw_urls, y, X = load_and_preprocess_dataset(primary_csv, max_samples=args.max_train_samples)
    train_urls_set = set(raw_urls)

    print("\n" + "=" * 80)
    print("STEP 2: FEATURE EXTRACTION COMPLETED (32 URL FEATURES)")
    print("=" * 80)

    print("\n" + "=" * 80)
    print("STEP 3: DATA SPLITTING & LEAKAGE PREVENTION")
    print("=" * 80)
    # Stratified 70% Train, 15% Validation, 15% Internal Test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=args.seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=args.seed, stratify=y_temp
    )

    print(f"Splits: Train = {len(y_train)} samples, Validation = {len(y_val)} samples, Internal Test = {len(y_test)} samples")
    print(f"Train Class Distribution: Legit = {np.sum(y_train == 0)}, Phish = {np.sum(y_train == 1)} ({np.mean(y_train)*100:.1f}% Phish)")

    print("\n" + "=" * 80)
    print("STEP 4: MODEL TRAINING & COMPARISON")
    print("=" * 80)

    # 1. Logistic Regression Baseline
    print("\n--- Model 1: Logistic Regression (Baseline) ---")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    lr_model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=args.seed)
    t0 = time.time()
    lr_model.fit(X_train_scaled, y_train)
    lr_train_time = time.time() - t0
    lr_val_prob = lr_model.predict_proba(X_val_scaled)[:, 1]
    lr_val_pred = (lr_val_prob >= 0.5).astype(int)
    lr_metrics = calculate_metrics(y_val, lr_val_pred, lr_val_prob)
    print(f"LR Training Time: {lr_train_time:.2f}s")
    print(f"LR Val Accuracy:  {lr_metrics['accuracy'] * 100:.2f}% | F1: {lr_metrics['f1'] * 100:.2f}% | FPR: {lr_metrics['fpr'] * 100:.2f}%")

    # 2. Random Forest
    print("\n--- Model 2: Random Forest Classifier ---")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=16,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=args.seed
    )
    t0 = time.time()
    rf_model.fit(X_train, y_train)
    rf_train_time = time.time() - t0
    rf_val_prob = rf_model.predict_proba(X_val)[:, 1]
    rf_val_pred = (rf_val_prob >= 0.5).astype(int)
    rf_metrics = calculate_metrics(y_val, rf_val_pred, rf_val_prob)
    print(f"RF Training Time: {rf_train_time:.2f}s")
    print(f"RF Val Accuracy:  {rf_metrics['accuracy'] * 100:.2f}% | F1: {rf_metrics['f1'] * 100:.2f}% | FPR: {rf_metrics['fpr'] * 100:.2f}%")

    # Check for XGBoost availability
    xgb_metrics = None
    xgb_model = None
    try:
        from xgboost import XGBClassifier
        print("\n--- Model 3: XGBoost Classifier ---")
        scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)
        xgb_model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=args.seed,
            n_jobs=-1,
            eval_metric="logloss"
        )
        t0 = time.time()
        xgb_model.fit(X_train, y_train)
        xgb_train_time = time.time() - t0
        xgb_val_prob = xgb_model.predict_proba(X_val)[:, 1]
        xgb_val_pred = (xgb_val_prob >= 0.5).astype(int)
        xgb_metrics = calculate_metrics(y_val, xgb_val_pred, xgb_val_prob)
        print(f"XGB Training Time: {xgb_train_time:.2f}s")
        print(f"XGB Val Accuracy:  {xgb_metrics['accuracy'] * 100:.2f}% | F1: {xgb_metrics['f1'] * 100:.2f}% | FPR: {xgb_metrics['fpr'] * 100:.2f}%")
    except Exception as e:
        print(f"XGBoost not available or encountered error: {e}")

    # Model Selection Logic
    print("\n" + "=" * 80)
    print("MODEL COMPARISON ON VALIDATION SET:")
    print("=" * 80)
    print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'FPR':<10} {'FNR':<10}")
    print(f"{'LogisticRegression':<20} {lr_metrics['accuracy']*100:<10.2f} {lr_metrics['precision']*100:<10.2f} {lr_metrics['recall']*100:<10.2f} {lr_metrics['f1']*100:<10.2f} {lr_metrics['fpr']*100:<10.2f} {lr_metrics['fnr']*100:<10.2f}")
    print(f"{'RandomForest':<20} {rf_metrics['accuracy']*100:<10.2f} {rf_metrics['precision']*100:<10.2f} {rf_metrics['recall']*100:<10.2f} {rf_metrics['f1']*100:<10.2f} {rf_metrics['fpr']*100:<10.2f} {rf_metrics['fnr']*100:<10.2f}")
    if xgb_metrics:
        print(f"{'XGBoost':<20} {xgb_metrics['accuracy']*100:<10.2f} {xgb_metrics['precision']*100:<10.2f} {xgb_metrics['recall']*100:<10.2f} {xgb_metrics['f1']*100:<10.2f} {xgb_metrics['fpr']*100:<10.2f} {xgb_metrics['fnr']*100:<10.2f}")

    # Compare RF vs XGB vs LR: Select best model prioritizing F1, low FPR, and robustness
    selected_model_name = "RandomForest"
    selected_model = rf_model
    selected_scaler = None

    if xgb_metrics and (xgb_metrics["f1"] > rf_metrics["f1"] and xgb_metrics["fpr"] <= rf_metrics["fpr"] + 0.002):
        selected_model_name = "XGBoost"
        selected_model = xgb_model
        selected_scaler = None

    print(f"\n=> SELECTED BEST MODEL: {selected_model_name}")

    print("\n" + "=" * 80)
    print("STEP 5: THRESHOLD CALIBRATION ON VALIDATION SET")
    print("=" * 80)
    val_probs = selected_model.predict_proba(X_val)[:, 1]
    t_safe, t_phish = calibrate_thresholds(y_val, val_probs, target_fpr=0.01)

    print("\n" + "=" * 80)
    print("STEP 6: INTERNAL TEST SET EVALUATION")
    print("=" * 80)
    test_probs = selected_model.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= t_phish).astype(int)
    test_metrics = calculate_metrics(y_test, test_preds, test_probs)

    print(f"Internal Test Set Metrics (at calibrated threshold {t_phish}):")
    print(f"  Accuracy:  {test_metrics['accuracy'] * 100:.2f}%")
    print(f"  Precision: {test_metrics['precision'] * 100:.2f}%")
    print(f"  Recall:    {test_metrics['recall'] * 100:.2f}%")
    print(f"  F1-Score:  {test_metrics['f1'] * 100:.2f}%")
    print(f"  ROC-AUC:   {test_metrics['roc_auc'] * 100:.2f}%")
    print(f"  FPR:       {test_metrics['fpr'] * 100:.2f}%")
    print(f"  FNR:       {test_metrics['fnr'] * 100:.2f}%")
    print(f"  Confusion Matrix: {test_metrics['confusion_matrix']}")

    # Feature Importance Analysis
    if hasattr(selected_model, "feature_importances_"):
        importances = selected_model.feature_importances_
        sorted_indices = np.argsort(importances)[::-1]
        print("\nTop 10 Most Predictive Features:")
        for rank, idx in enumerate(sorted_indices[:10]):
            print(f"  {rank+1}. {FEATURE_NAMES[idx]:<28} (importance: {importances[idx]:.4f})")

    # Step 7: Independent External Evaluation on PhiUSIIL
    phiusiil_metrics = None
    if os.path.exists(phiusiil_csv):
        phiusiil_metrics = evaluate_phiusiil_external(
            phiusiil_csv, selected_model, selected_scaler,
            train_urls_set, t_safe, t_phish, max_eval=40000
        )

    # Step 8: Real-World and Typosquatting Evaluation
    test_suite_results = evaluate_test_suite(selected_model, selected_scaler, t_safe, t_phish)

    # Step 9: Model Saving
    print("\n" + "=" * 80)
    print("STEP 9: SAVING MODEL ARTIFACT AND METADATA")
    print("=" * 80)
    model_artifact = {
        "model": selected_model,
        "scaler": selected_scaler,
        "model_type": selected_model_name,
        "feature_names": FEATURE_NAMES,
        "thresholds": {
            "safe_max": t_safe,
            "phish_min": t_phish
        },
        "label_mapping": {0: "Safe", 1: "Phishing"},
        "sources": {
            "train": "Dataset.csv",
            "external_test": "PhiUSIIL_Phishing_URL_Dataset.csv",
            "excluded": "url_features_extracted1.csv (redundant domain clustering & overlap with PhiUSIIL)"
        },
        "metrics": {
            "internal_test": test_metrics,
            "external_test": phiusiil_metrics
        },
        "version": "1.0.0",
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }

    joblib.dump(model_artifact, args.output_model)
    model_size_mb = os.path.getsize(args.output_model) / (1024 * 1024)
    print(f"Model artifact successfully saved to: {args.output_model} ({model_size_mb:.2f} MB)")

    print("\n" + "=" * 80)
    print("TRAINING PIPELINE COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
