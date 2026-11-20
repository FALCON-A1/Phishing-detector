"""
Spam vs Ham Detector Module for English and Arabic.
Provides text preprocessing, heuristic spam/scam pattern detection,
and machine learning inference with explainable reasons and protection advice.
"""

import os
import re
import sys
import time
from typing import Dict, List, Tuple, Any, Optional
import joblib
import numpy as np

# Current directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Global model container
SPAM_MODEL_STATE: Dict[str, Any] = {
    "loaded": False,
    "vectorizer": None,
    "model": None,
    "thresholds": {"ham_max": 0.35, "spam_min": 0.65},
    "version": "1.0.0"
}

# Arabic diacritics / tashkeel regex
ARABIC_TASHKEEL = re.compile(r"[\u0617-\u061A\u064B-\u0652]")

# Common Scam / Spam Trigger Dictionaries
TRIGGERS_EN = {
    "urgency": [
        "urgent", "immediately", "account suspended", "suspended", "action required",
        "verify your", "security alert", "locked", "expire", "24 hours", "final notice",
        "unauthorized", "compromised", "reactivate", "restricted"
    ],
    "prize_lottery": [
        "congratulations", "winner", "you have won", "claim prize", "free gift",
        "cash prize", "awarded", "selected to receive", "voucher", "guaranteed",
        "claim your", "claim code", "won a", "$1,000", "£1,000", "gift card"
    ],
    "financial_scam": [
        "wire transfer", "inheritance", "cryptocurrency", "bitcoin", "investment return",
        "passive income", "double your money", "no risk", "instant cash", "loan approved"
    ],
    "contact_diversion": [
        "call now", "call 0800", "call 090", "text to", "reply stop", "whatsapp me",
        "click here", "tap link", "contact customer support"
    ]
}

TRIGGERS_AR = {
    "urgency": [
        "عاجل", "تم ايقاف حسابك", "ايقاف الحساب", "تجميد حسابك", "تنبيه امني",
        "يرجى التحديث", "تحديث بياناتك", "خلال 24 ساعة", "فورا", "معلق", "انذار اخير"
    ],
    "prize_lottery": [
        "مبروك", "ربحت", "فزت معنا", "جائزة مالية", "استلم جائزتك", "سحب على",
        "سيارة مجانا", "مكافأة مالية", "هدية مجانية", "ربح فوري", "مليون ريال"
    ],
    "financial_scam": [
        "تسديد قروض", "تداول مضمون", "ارباح يومية", "استثمار مضمون", "استرجاع اموال",
        "بدون كفيل", "تسهيلات بنكية", "عمل من المنزل براتب"
    ],
    "suspicious_services": [
        "شهاده ايلتس", "شهادات معتمده بدون اختبار", "بدون اختبارات", "تسريب اختبارات",
        "تعديل درجات", "سكس", "مقاطع", "معصيتي", "تعاالي خاص"
    ],
    "contact_diversion": [
        "تواصل واتساب", "راسلنا واتس", "تواصل عبر الواتس", "اضغط على الرابط",
        "راسلني خاص", "رابط التسجيل", "حمل التطبيق قبل الحذف"
    ]
}


def preprocess_text(text: str) -> str:
    """
    Unified text preprocessor for both English and Arabic.
    Normalizes Arabic orthography (alif, yaa, taa marbuta) and strips diacritics.
    Lowercases English and strips extraneous whitespace.
    """
    if not isinstance(text, str):
        return ""
    text = text.strip()

    # Normalize Arabic diacritics and tatweel
    text = re.sub(ARABIC_TASHKEEL, "", text)
    text = re.sub(r"\u0640", "", text)
    # Normalize Alef variations
    text = re.sub(r"[إأآا]", "ا", text)
    # Normalize Yaa
    text = re.sub(r"ى", "ي", text)
    # Normalize Taa Marbuta
    text = re.sub(r"ة", "ه", text)

    return text.lower()


def detect_language(text: str) -> str:
    """Detect if message is English, Arabic, or Mixed."""
    arabic_chars = len(re.findall(r"[\u0600-\u06FF]", text))
    latin_chars = len(re.findall(r"[a-zA-Z]", text))

    if arabic_chars > 0 and latin_chars > 0 and min(arabic_chars, latin_chars) >= 4:
        return "Mixed (Arabic & English)"
    elif arabic_chars > latin_chars:
        return "Arabic"
    else:
        return "English"


def extract_spam_indicators(text: str) -> Dict[str, Any]:
    """Extract explainable heuristic scam and spam indicators."""
    lower_text = text.lower()
    norm_text = preprocess_text(text)

    reasons: List[str] = []
    matched_triggers: List[str] = []

    # 1. URL links and shorteners
    has_links = bool(re.search(r"https?://\S+|www\.\S+|bit\.ly/\S+|t\.co/\S+|tinyurl\.com/\S+", text))
    if has_links:
        reasons.append("Contains unverified web link or URL shortener (يحتوي على رابط إلكتروني)")

    # 2. Phone numbers or premium call instructions
    has_phone = bool(re.search(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4,6}\b", text))
    if has_phone and any(k in lower_text for k in ["call", "contact", "text", "اتصل", "تواصل"]):
        reasons.append("Prompts recipient to call or text an unverified number (طلب اتصال برقم غير موثق)")

    # 3. English Triggers
    for category, keywords in TRIGGERS_EN.items():
        for kw in keywords:
            if kw in lower_text:
                matched_triggers.append(kw)
                if category == "urgency" and "Contains urgent psychological pressure" not in reasons:
                    reasons.append(f"Contains urgent psychological pressure / account threat ('{kw}')")
                elif category == "prize_lottery" and "Promises lottery, prize, or reward" not in reasons:
                    reasons.append(f"Promises unrealistic lottery, reward, or financial prize ('{kw}')")
                elif category == "financial_scam" and "Contains high-risk financial / investment offer" not in reasons:
                    reasons.append(f"Contains high-risk financial / investment scheme ('{kw}')")
                elif category == "contact_diversion" and "Directs user to contact or click externally" not in reasons:
                    reasons.append(f"Directs user to external action ('{kw}')")

    # 4. Arabic Triggers
    for category, keywords in TRIGGERS_AR.items():
        for kw in keywords:
            if kw in norm_text:
                matched_triggers.append(kw)
                if category == "urgency" and "يحتوي على تهديد أو استعجال بتجميد الحساب" not in reasons:
                    reasons.append(f"يحتوي على تهديد أو استعجال بتجميد الحساب ('{kw}')")
                elif category == "prize_lottery" and "إيهام بالفوز بجوائز أو مكاسب مالية وهمية" not in reasons:
                    reasons.append(f"إيهام بالفوز بجوائز أو مكاسب مالية وهمية ('{kw}')")
                elif category == "financial_scam" and "عروض مالية أو تسديد قروض غير موثقة" not in reasons:
                    reasons.append(f"عروض مالية أو تسديد قروض غير موثقة ('{kw}')")
                elif category == "suspicious_services" and "ترويج لخدمات احتيالية أو محتوى محظور" not in reasons:
                    reasons.append(f"ترويج لخدمات احتيالية أو محتوى غير لائق ('{kw}')")
                elif category == "contact_diversion" and "طلب التواصل عبر واتساب أو روابط خارجية" not in reasons:
                    reasons.append(f"طلب التواصل عبر واتساب أو روابط خارجية ('{kw}')")

    # 5. Excessive capitalization or exclamation marks
    if len(text) > 20 and sum(1 for c in text if c.isupper()) / max(len(re.findall(r"[a-zA-Z]", text)), 1) > 0.6:
        reasons.append("Excessive uppercase lettering indicates aggressive promotional messaging")
    if text.count("!") >= 3 or text.count("؟") >= 3:
        reasons.append("Excessive exclamation/punctuation often used in social engineering")

    return {
        "has_links": has_links,
        "matched_triggers": matched_triggers,
        "reasons": reasons,
        "trigger_count": len(matched_triggers)
    }


def load_spam_model(model_path: str = "spam_model.joblib") -> bool:
    """Load spam vectorizer and classifier from disk."""
    full_path = os.path.join(CURRENT_DIR, model_path) if not os.path.isabs(model_path) else model_path
    if not os.path.exists(full_path):
        print(f"Warning: Spam model file not found at {full_path}")
        return False

    try:
        artifact = joblib.load(full_path)
        SPAM_MODEL_STATE["vectorizer"] = artifact["vectorizer"]
        SPAM_MODEL_STATE["model"] = artifact["model"]
        SPAM_MODEL_STATE["thresholds"] = artifact.get("thresholds", {"ham_max": 0.35, "spam_min": 0.65})
        SPAM_MODEL_STATE["version"] = artifact.get("version", "1.0.0")
        SPAM_MODEL_STATE["loaded"] = True
        print(f"Successfully loaded Spam model: {type(artifact['model']).__name__}")
        return True
    except Exception as e:
        print(f"Error loading spam model: {e}")
        SPAM_MODEL_STATE["loaded"] = False
        return False


def get_security_advice(prediction: str, lang: str) -> str:
    """Generate protective guidance to keep users safe from scams."""
    is_ar = "Arabic" in lang

    if prediction == "Spam":
        if is_ar:
            return "⚠️ تنبيه أمني: هذه الرسالة تتضمن مؤشرات احتيال أو سبام عالية. لا تقم بالضغط على أي روابط، ولا تشارك أي بيانات بنكية أو رموز تحقق (OTP). قم بحظر وحذف الرسالة فوراً."
        else:
            return "⚠️ Security Alert: This message shows strong characteristics of a phishing/spam scam. Do not click links, disclose OTP codes, or contact the sender. Delete and report the message."
    elif prediction == "Suspicious":
        if is_ar:
            return "⚡ تحذير: الرسالة تحتوي على صياغة مشبوهة أو عناصر غير معتادة. تحقق من هوية المرسل عبر القنوات الرسمية قبل اتخاذ أي إجراء."
        else:
            return "⚡ Caution: This message contains ambiguous or promotional patterns. Verify the sender's identity through official channels before responding or clicking any links."
    else:
        if is_ar:
            return "✅ رسالة آمنة: لم يتم رصد أي أنماط احتيالية أو سبام معروفة في نص هذه الرسالة."
        else:
            return "✅ Safe: No recognized malicious, deceptive, or spam patterns detected in this message."


def classify_message(text: str) -> Dict[str, Any]:
    """
    Classify a message text into Spam, Suspicious, or Ham.
    Returns prediction, calibrated risk score, confidence, language, reasons, and security advice.
    """
    clean_text = text.strip()
    if not clean_text:
        return {
            "text": "",
            "prediction": "Ham",
            "confidence": 1.0,
            "risk_score": 0.0,
            "language": "Unknown",
            "reasons": ["Empty message provided."],
            "security_advice": "No content to evaluate."
        }

    lang = detect_language(clean_text)
    indicators = extract_spam_indicators(clean_text)
    reasons = indicators["reasons"]
    has_signals = len(reasons) > 0

    # Model probability
    if SPAM_MODEL_STATE["loaded"] and SPAM_MODEL_STATE["model"] is not None:
        vec = SPAM_MODEL_STATE["vectorizer"]
        model = SPAM_MODEL_STATE["model"]
        X = vec.transform([clean_text])
        if hasattr(model, "predict_proba"):
            raw_prob = float(model.predict_proba(X)[0, 1])
        else:
            df = float(model.decision_function(X)[0])
            raw_prob = 1.0 / (1.0 + np.exp(-df))
    else:
        # Fallback to heuristic scoring if model is not yet loaded
        raw_prob = min(0.20 + 0.25 * indicators["trigger_count"] + (0.30 if indicators["has_links"] else 0.0), 0.95)

    # 3-Level Calibrated Decision
    t_ham = SPAM_MODEL_STATE["thresholds"].get("ham_max", 0.35)
    t_spam = SPAM_MODEL_STATE["thresholds"].get("spam_min", 0.65)

    # Decision logic
    if raw_prob >= t_spam or (raw_prob >= 0.50 and indicators["trigger_count"] >= 2) or (indicators["trigger_count"] >= 3):
        prediction = "Spam"
        calibrated_risk = max(raw_prob, 0.85 if has_signals else 0.75)
        confidence = max(raw_prob, 0.88)
        if not reasons:
            reasons.append("Statistical text characteristics match known spam campaigns.")
    elif (raw_prob >= 0.60) or (raw_prob >= t_ham and has_signals) or (indicators["trigger_count"] >= 1):
        prediction = "Suspicious"
        calibrated_risk = max(raw_prob, 0.55)
        confidence = 0.80
        if not reasons:
            reasons.append("Moderate similarity to unsolicited or promotional messages.")
    else:
        prediction = "Ham"
        calibrated_risk = min(raw_prob, 0.15)
        confidence = 1.0 - raw_prob
        if not reasons:
            reasons.append("Natural conversational or authentic message structure.")

    advice = get_security_advice(prediction, lang)

    return {
        "text": clean_text,
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "risk_score": round(calibrated_risk, 4),
        "language": lang,
        "reasons": reasons,
        "security_advice": advice
    }


if __name__ == "__main__":
    load_spam_model()
    test_msgs = [
        "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121",
        "Hello mom, can you pick me up from school today?",
        "مبروك ربحت معنا 50000 ريال استلم جائزتك الان عبر الرابط",
        "السلام عليكم يا دكتور، هل تم تحديد موعد الامتحان القادم؟"
    ]
    for m in test_msgs:
        res = classify_message(m)
        print(f"\n[{res['prediction']:10}] ({res['language']}) Risk: {res['risk_score']*100:.1f}%")
        print(f"Message: {m}")
        print(f"Reasons: {res['reasons']}")
        print(f"Advice:  {res['security_advice']}")
