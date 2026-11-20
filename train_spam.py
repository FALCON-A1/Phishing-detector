"""
Training Script for Multilingual (English & Arabic) Spam vs Ham Detector.
Trains a lightweight, high-performance TF-IDF + Logistic Regression pipeline
on English SMS spam and Arabic tweets/messages datasets.
"""

import os
import sys
import time
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score

# Ensure root directory is accessible
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from spam_detector import preprocess_text


def load_datasets(dataset_dir: str):
    """Load and format both English and Arabic datasets."""
    spam_csv = os.path.join(dataset_dir, "spam.csv")
    arabic_xlsx = os.path.join(dataset_dir, "Dataset of Arabic Spam and Ham Tweets.xlsx")

    if not os.path.exists(spam_csv) or not os.path.exists(arabic_xlsx):
        raise FileNotFoundError(f"Missing dataset files in {dataset_dir}")

    # 1. Load English SMS dataset
    try:
        df_en = pd.read_csv(spam_csv, encoding="utf-8")
    except UnicodeDecodeError:
        df_en = pd.read_csv(spam_csv, encoding="latin-1")

    df_en = df_en[["v1", "v2"]].dropna()
    df_en.columns = ["label", "text"]
    df_en["label"] = df_en["label"].str.lower().map({"ham": 0, "spam": 1})
    df_en["lang"] = "en"
    print(f"Loaded English Dataset: {len(df_en)} samples ({df_en['label'].sum()} spam, {(df_en['label'] == 0).sum()} ham)")

    # 2. Load Arabic Tweets/Messages dataset
    df_ar = pd.read_excel(arabic_xlsx)
    text_col = "Tweet Text" if "Tweet Text" in df_ar.columns else df_ar.columns[4]
    label_col = "Label" if "Label" in df_ar.columns else df_ar.columns[-1]

    df_ar = df_ar[[label_col, text_col]].dropna()
    df_ar.columns = ["label", "text"]
    df_ar["label"] = df_ar["label"].astype(str).str.strip().map({"Ham": 0, "Spam": 1, "ham": 0, "spam": 1})
    df_ar = df_ar.dropna(subset=["label"])
    df_ar["label"] = df_ar["label"].astype(int)
    df_ar["lang"] = "ar"
    print(f"Loaded Arabic Dataset:  {len(df_ar)} samples ({df_ar['label'].sum()} spam, {(df_ar['label'] == 0).sum()} ham)")

    # 3. Combine into unified corpus
    df_comb = pd.concat([df_en, df_ar], ignore_index=True)
    print(f"Total Unified Corpus:   {len(df_comb)} samples ({df_comb['label'].sum()} spam, {(df_comb['label'] == 0).sum()} ham)")
    return df_comb


def train_spam_model(dataset_dir: str = None, output_model_path: str = "spam_model.joblib"):
    """Train the multilingual spam detector and save model artifact."""
    if dataset_dir is None:
        dataset_dir = os.path.join(CURRENT_DIR, "Datasets", "Spam vs Ham Detector")

    print("\n" + "=" * 60)
    print("🚀 TRAINING MULTILINGUAL SPAM VS HAM DETECTOR")
    print("=" * 60)

    start_time = time.time()
    df = load_datasets(dataset_dir)

    # Stratified Train/Test split (80% train, 20% test)
    tr, te = train_test_split(df, test_size=0.20, random_state=42, stratify=df["label"])
    print(f"\nTraining set size: {len(tr)} | Test set size: {len(te)}")

    # Vectorization with custom Arabic+English preprocessor
    print("\nFitting TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(
        preprocessor=preprocess_text,
        ngram_range=(1, 2),
        max_features=35000,
        sublinear_tf=True
    )
    X_train = vectorizer.fit_transform(tr["text"])
    y_train = tr["label"].values

    # Logistic Regression with balanced class weights
    print("Training Balanced Logistic Regression model...")
    classifier = LogisticRegression(class_weight="balanced", C=5.0, max_iter=1000, random_state=42)
    classifier.fit(X_train, y_train)

    # Evaluation
    print("\n" + "-" * 50)
    print("📊 EVALUATION RESULTS ON INDEPENDENT TEST SPLIT")
    print("-" * 50)

    # 1. Overall Combined Test
    X_test_all = vectorizer.transform(te["text"])
    y_test_all = te["label"].values
    pred_all = classifier.predict(X_test_all)
    print(f"Overall Accuracy:  {accuracy_score(y_test_all, pred_all)*100:.2f}%")
    print(f"Overall Precision: {precision_score(y_test_all, pred_all)*100:.2f}%")
    print(f"Overall Recall:    {recall_score(y_test_all, pred_all)*100:.2f}%")
    print(f"Overall F1-Score:  {f1_score(y_test_all, pred_all)*100:.2f}%\n")

    # 2. English Sub-Evaluation
    te_en = te[te["lang"] == "en"]
    X_test_en = vectorizer.transform(te_en["text"])
    pred_en = classifier.predict(X_test_en)
    print("🇬🇧 English Test Metrics:")
    print(f"  Accuracy:  {accuracy_score(te_en['label'], pred_en)*100:.2f}%")
    print(f"  Precision: {precision_score(te_en['label'], pred_en)*100:.2f}%")
    print(f"  Recall:    {recall_score(te_en['label'], pred_en)*100:.2f}%")
    print(f"  F1-Score:  {f1_score(te_en['label'], pred_en)*100:.2f}%\n")

    # 3. Arabic Sub-Evaluation
    te_ar = te[te["lang"] == "ar"]
    X_test_ar = vectorizer.transform(te_ar["text"])
    pred_ar = classifier.predict(X_test_ar)
    print("🇸🇦 Arabic Test Metrics:")
    print(f"  Accuracy:  {accuracy_score(te_ar['label'], pred_ar)*100:.2f}%")
    print(f"  Precision: {precision_score(te_ar['label'], pred_ar)*100:.2f}%")
    print(f"  Recall:    {recall_score(te_ar['label'], pred_ar)*100:.2f}%")
    print(f"  F1-Score:  {f1_score(te_ar['label'], pred_ar)*100:.2f}%\n")

    # Train on full corpus for production deployment
    print("Training final model on full combined dataset...")
    X_full = vectorizer.fit_transform(df["text"])
    classifier.fit(X_full, df["label"].values)

    # Save artifact
    artifact = {
        "vectorizer": vectorizer,
        "model": classifier,
        "thresholds": {"ham_max": 0.35, "spam_min": 0.65},
        "version": "1.0.0",
        "trained_samples": len(df),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }

    output_path = os.path.join(CURRENT_DIR, output_model_path) if not os.path.isabs(output_model_path) else output_model_path
    joblib.dump(artifact, output_path, compress=3)
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)

    elapsed = time.time() - start_time
    print(f"✅ Saved spam model artifact to: {output_path} ({file_size_mb:.2f} MB)")
    print(f"⏱️ Total training completed in {elapsed:.2f} seconds")
    return output_path


if __name__ == "__main__":
    train_spam_model()
