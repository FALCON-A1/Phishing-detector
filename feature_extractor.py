"""
Feature Extraction Module for URL Phishing Detector.
Extracts lexical, structural, security, and typosquatting features from raw URLs.
Uses pure Python standard library for maximum portability and high inference speed.
"""

import math
import re
import urllib.parse
from collections import Counter
from typing import Dict, List, Tuple, Any

# Top targeted brands commonly spoofed in phishing attacks
TARGET_BRANDS = [
    "google", "apple", "microsoft", "amazon", "paypal", "facebook",
    "netflix", "yahoo", "twitter", "linkedin", "instagram", "dropbox",
    "github", "whatsapp", "telegram", "chase", "wellsfargo", "bankofamerica",
    "citi", "binance", "coinbase", "metamask", "steam", "spotify",
    "roblox", "tiktok", "outlook", "icloud", "adobe", "walmart",
    "ebay", "dhl", "fedex", "usps"
]

# High-risk TLDs frequently abused in malicious campaigns
SUSPICIOUS_TLDS = {
    "tk", "ml", "ga", "cf", "gq", "top", "xyz", "work", "click", "fit",
    "buzz", "loan", "surf", "icu", "rest", "hair", "kim", "vip", "monster",
    "country", "stream", "gdn", "mom", "rodeo", "site", "online", "club"
}

# Suspicious keywords commonly seen in credential phishing / social engineering
SUSPICIOUS_KEYWORDS = [
    "login", "signin", "verify", "verification", "account", "banking",
    "secure", "security", "update", "confirm", "confirmation", "wallet",
    "admin", "service", "support", "password", "credential", "auth",
    "recover", "billing", "ebayisapi", "webscr", "cmd", "session",
    "suspended", "unlock", "alert", "validate", "renew"
]

# Suspicious executable or script file extensions
SUSPICIOUS_EXTENSIONS = (
    ".exe", ".apk", ".scr", ".bat", ".cmd", ".iso", ".zip", ".vbs",
    ".msi", ".hta", ".jar", ".ps1", ".sh", ".rar", ".7z"
)

# Common Cyrillic / Greek homoglyphs mimicking Latin letters
HOMOGLYPH_MAP = {
    "\u0430": "a", "\u0441": "c", "\u0435": "e", "\u043e": "o", "\u0440": "p",
    "\u0455": "s", "\u0445": "x", "\u0443": "y", "\u0456": "i", "\u0458": "j",
    "\u03bf": "o", "\u03c1": "p", "\u03b1": "a", "\u03bd": "v"
}

# Leetspeak / visual character substitution mapping
LEET_TRANS = str.maketrans({
    "0": "o",
    "1": "l",
    "3": "e",
    "4": "a",
    "5": "s",
    "8": "b",
    "@": "a",
    "$": "s",
    "!": "i"
})


def calculate_entropy(text: str) -> float:
    """Calculate Shannon entropy of a text string."""
    if not text:
        return 0.0
    length = len(text)
    counts = Counter(text)
    return -sum((cnt / length) * math.log2(cnt / length) for cnt in counts.values())


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate minimum edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def normalize_substitutions(text: str) -> str:
    """Normalize common typosquatting substitutions (e.g., 0->o, 1->l, rn->m, vv->w)."""
    text = text.lower()
    text = text.translate(LEET_TRANS)
    text = text.replace("rn", "m").replace("vv", "w")
    return text


def parse_url_components(url: str) -> Tuple[str, str, str, str, str, str]:
    """
    Safely parse URL into components:
    (scheme, full_domain, sld, tld, path, query)
    """
    url_stripped = url.strip()
    if not url_stripped.startswith(("http://", "https://", "ftp://")):
        url_parsed = urllib.parse.urlparse("http://" + url_stripped)
        scheme = ""
    else:
        url_parsed = urllib.parse.urlparse(url_stripped)
        scheme = url_parsed.scheme.lower()

    netloc = url_parsed.netloc.lower()
    # Strip port if present
    if ":" in netloc and not netloc.endswith("]"):
        full_domain = netloc.split(":")[0]
    else:
        full_domain = netloc

    # Extract domain parts
    domain_parts = full_domain.split(".")
    if len(domain_parts) >= 2:
        tld = domain_parts[-1]
        sld = domain_parts[-2]
        # Handle compound TLDs like co.uk, com.au, edu.eg
        if len(domain_parts) >= 3 and domain_parts[-2] in ("co", "com", "edu", "gov", "org", "net"):
            tld = f"{domain_parts[-2]}.{domain_parts[-1]}"
            sld = domain_parts[-3]
    else:
        sld = full_domain
        tld = ""

    path = url_parsed.path or ""
    query = url_parsed.query or ""

    return scheme, full_domain, sld, tld, path, query


def is_ip_address(domain: str) -> bool:
    """Check if domain is an IPv4 or IPv6 address."""
    # IPv4 regex
    ipv4_pattern = r"^(\d{1,3}\.){3}\d{1,3}$"
    if re.match(ipv4_pattern, domain):
        parts = domain.split(".")
        return all(0 <= int(p) <= 255 for p in parts)
    # Basic IPv6 check
    if ":" in domain or domain.startswith("["):
        return True
    return False


def detect_homoglyphs(domain: str) -> int:
    """Count non-ASCII homoglyphs resembling Latin letters."""
    count = 0
    for char in domain:
        if char in HOMOGLYPH_MAP or ord(char) > 127:
            count += 1
    return count


def max_char_repetition(text: str) -> int:
    """Find maximum consecutive identical character repetition."""
    if not text:
        return 0
    max_rep = 1
    cur_rep = 1
    for i in range(1, len(text)):
        if text[i] == text[i - 1]:
            cur_rep += 1
            if cur_rep > max_rep:
                max_rep = cur_rep
        else:
            cur_rep = 1
    return max_rep


def extract_typosquatting_indicators(full_domain: str, sld: str, path: str) -> Dict[str, Any]:
    """
    Extract typosquatting, brand impersonation, and homoglyph metrics.
    """
    norm_sld = normalize_substitutions(sld)
    has_punycode = 1 if "xn--" in full_domain else 0
    homoglyph_count = detect_homoglyphs(full_domain)
    # Strip standard www. prefix to avoid measuring www as 3-char repetition
    clean_dom = full_domain[4:] if full_domain.startswith("www.") else full_domain
    max_rep = max_char_repetition(clean_dom)

    min_edit_dist = 99
    closest_brand = ""
    is_brand_lookalike = 0
    brand_in_sub_or_path = 0

    lower_full = full_domain.lower()
    lower_path = path.lower()

    for brand in TARGET_BRANDS:
        # Check brand in subdomain or path when the registered domain is NOT that brand
        if brand in lower_full or brand in lower_path:
            if sld != brand:
                brand_in_sub_or_path = 1

        # Check lookalike distance
        dist = levenshtein_distance(sld, brand)
        if dist < min_edit_dist:
            min_edit_dist = dist
            closest_brand = brand

        # Check leetspeak match (e.g. paypa1 -> paypal, amaz0n -> amazon)
        if sld != brand:
            if norm_sld == brand:
                is_brand_lookalike = 1
            elif len(sld) >= 4 and dist <= 1:
                is_brand_lookalike = 1
            elif len(sld) >= 6 and dist == 2 and not sld.startswith(brand):
                is_brand_lookalike = 1

    return {
        "min_brand_edit_distance": float(min_edit_dist),
        "is_brand_lookalike": float(is_brand_lookalike),
        "brand_in_subdomain_or_path": float(brand_in_sub_or_path),
        "char_repetition_max": float(max_rep),
        "has_punycode": float(has_punycode),
        "has_homoglyphs": float(1 if homoglyph_count > 0 else 0)
    }


def extract_features(url: str) -> Dict[str, float]:
    """
    Extract all URL features as an ordered dictionary of float values.
    Identical feature extraction function for training and production API.
    """
    raw_url = str(url).strip()
    url_len = len(raw_url)
    scheme, full_domain, sld, tld, path, query = parse_url_components(raw_url)

    dom_len = len(full_domain)
    path_len = len(path)
    query_len = len(query)

    # Character counts
    dot_cnt = raw_url.count(".")
    dot_dom_cnt = full_domain.count(".")
    dash_cnt = raw_url.count("-")
    dash_dom_cnt = full_domain.count("-")
    digit_cnt = sum(c.isdigit() for c in raw_url)
    digit_dom_cnt = sum(c.isdigit() for c in full_domain)

    # Special characters
    special_chars = set("@?=_%&~#+;:$*!,")
    special_cnt = sum(1 for c in raw_url if c in special_chars)

    # Subdomains count (labels excluding sld and tld)
    parts = [p for p in full_domain.split(".") if p]
    subdom_cnt = max(0, len(parts) - 2) if len(parts) > 2 else 0

    # Query parameters
    param_cnt = len(query.split("&")) if query else 0

    # Security indicators
    is_https = 1.0 if scheme == "https" else 0.0
    has_ip = 1.0 if is_ip_address(full_domain) else 0.0

    lower_url = raw_url.lower()
    keyword_cnt = sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in lower_url)

    has_encoding = 1.0 if "%" in raw_url else 0.0
    url_entropy = calculate_entropy(raw_url)
    dom_entropy = calculate_entropy(full_domain)

    digit_ratio = digit_cnt / max(url_len, 1)
    dom_digit_ratio = digit_dom_cnt / max(dom_len, 1)

    excessive_subdomains = 1.0 if subdom_cnt >= 3 else 0.0
    has_at_symbol = 1.0 if "@" in raw_url else 0.0
    has_double_slash = 1.0 if ("//" in path or "//" in query) else 0.0
    suspicious_tld_flag = 1.0 if tld.lower() in SUSPICIOUS_TLDS else 0.0
    suspicious_ext_flag = 1.0 if path.lower().endswith(SUSPICIOUS_EXTENSIONS) else 0.0

    # Typosquatting and impersonation indicators
    typo_info = extract_typosquatting_indicators(full_domain, sld, path)

    features = {
        "url_length": float(url_len),
        "domain_length": float(dom_len),
        "path_length": float(path_len),
        "query_length": float(query_len),
        "num_dots": float(dot_cnt),
        "num_dots_domain": float(dot_dom_cnt),
        "num_hyphens": float(dash_cnt),
        "num_hyphens_domain": float(dash_dom_cnt),
        "num_digits": float(digit_cnt),
        "num_digits_domain": float(digit_dom_cnt),
        "num_special_chars": float(special_cnt),
        "num_subdomains": float(subdom_cnt),
        "num_params": float(param_cnt),
        "is_https": is_https,
        "has_ip": has_ip,
        "suspicious_keyword_count": float(keyword_cnt),
        "has_url_encoding": has_encoding,
        "url_entropy": round(url_entropy, 4),
        "domain_entropy": round(dom_entropy, 4),
        "digit_ratio": round(digit_ratio, 4),
        "domain_digit_ratio": round(dom_digit_ratio, 4),
        "excessive_subdomains": excessive_subdomains,
        "has_at_symbol": has_at_symbol,
        "has_double_slash_redirect": has_double_slash,
        "suspicious_tld": suspicious_tld_flag,
        "suspicious_file_ext": suspicious_ext_flag,
        "min_brand_edit_distance": typo_info["min_brand_edit_distance"],
        "is_brand_lookalike": typo_info["is_brand_lookalike"],
        "brand_in_subdomain_or_path": typo_info["brand_in_subdomain_or_path"],
        "char_repetition_max": typo_info["char_repetition_max"],
        "has_punycode": typo_info["has_punycode"],
        "has_homoglyphs": typo_info["has_homoglyphs"]
    }

    return features


FEATURE_NAMES = list(extract_features("https://example.com").keys())


def extract_features_vector(url: str, feature_names: List[str] = None) -> List[float]:
    """Extract features as an ordered numeric list matching feature_names order."""
    if feature_names is None:
        feature_names = FEATURE_NAMES
    feat_dict = extract_features(url)
    return [feat_dict[name] for name in feature_names]


if __name__ == "__main__":
    test_urls = [
        "https://www.google.com",
        "https://go0g1e.com/login",
        "https://paypa1.com/verify-account",
        "http://192.168.1.1/admin/login.php",
        "https://paypal.com.account-verification.xyz/login",
        "https://en.wikipedia.org/wiki/Phishing"
    ]
    print(f"Total features: {len(FEATURE_NAMES)}")
    print(f"Feature names: {FEATURE_NAMES}")
    for u in test_urls:
        res = extract_features(u)
        print(f"\nURL: {u}")
        print(f"  lookalike: {res['is_brand_lookalike']}, brand_in_sub: {res['brand_in_subdomain_or_path']}, min_dist: {res['min_brand_edit_distance']}, kw_cnt: {res['suspicious_keyword_count']}")
