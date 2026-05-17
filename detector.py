"""
Phishing Email Detector
Author: Patricia Naamala
Description: Uses machine learning (Naive Bayes) to classify emails as phishing or legitimate.
             Extracts features like urgency words, suspicious links, sender patterns, and more.
"""

import re
import math
from collections import defaultdict


# ──────────────────────────────────────────────
# Feature Extraction
# ──────────────────────────────────────────────

URGENCY_WORDS = [
    "urgent", "immediately", "account suspended", "verify now", "action required",
    "limited time", "act now", "expire", "click here", "confirm your",
    "update your", "unusual activity", "unauthorized", "suspended", "locked",
    "alert", "warning", "important notice", "final notice", "last chance"
]

SUSPICIOUS_PHRASES = [
    "dear customer", "dear user", "dear account holder", "valued customer",
    "your account has been", "we have noticed", "please verify", "click the link below",
    "do not ignore", "failure to verify", "we detected", "kindly update"
]

TRUSTED_DOMAINS = [
    "gmail.com", "yahoo.com", "outlook.com", "hotmail.com",
    "safaricom.co.ke", "equity.co.ke", "kcbgroup.com"
]


def extract_features(email: dict) -> dict:
    """Extract numerical features from an email dictionary."""
    subject = email.get("subject", "").lower()
    body = email.get("body", "").lower()
    sender = email.get("sender", "").lower()
    full_text = subject + " " + body

    features = {}

    # 1. Urgency word count
    features["urgency_count"] = sum(1 for w in URGENCY_WORDS if w in full_text)

    # 2. Suspicious phrase count
    features["suspicious_phrases"] = sum(1 for p in SUSPICIOUS_PHRASES if p in full_text)

    # 3. Number of URLs in body
    urls = re.findall(r'http[s]?://\S+', body)
    features["url_count"] = len(urls)

    # 4. Suspicious URL patterns (IP addresses, URL shorteners, misleading domains)
    suspicious_url_patterns = [
        r'http[s]?://\d+\.\d+\.\d+\.\d+',   # IP-based URL
        r'bit\.ly|tinyurl|goo\.gl|t\.co',     # URL shorteners
        r'secure.*login|login.*secure',        # fake secure pages
        r'paypa1|arnazon|g00gle|micros0ft'     # typosquatting
    ]
    features["suspicious_urls"] = sum(
        1 for url in urls
        for pattern in suspicious_url_patterns
        if re.search(pattern, url)
    )

    # 5. Sender domain trust
    sender_domain_match = re.search(r'@([\w\.\-]+)', sender)
    if sender_domain_match:
        domain = sender_domain_match.group(1)
        features["trusted_sender"] = 1 if domain in TRUSTED_DOMAINS else 0
        features["sender_domain_length"] = len(domain)
        # Suspicious: domain has many subdomains
        features["sender_subdomain_count"] = domain.count(".")
    else:
        features["trusted_sender"] = 0
        features["sender_domain_length"] = 0
        features["sender_subdomain_count"] = 0

    # 6. Excessive punctuation / caps (common in phishing)
    features["exclamation_count"] = full_text.count("!")
    features["caps_ratio"] = sum(1 for c in (subject + " " + email.get("body", "")) if c.isupper()) / max(len(full_text), 1)

    # 7. Requests for personal info
    personal_info_words = ["password", "ssn", "social security", "credit card", "bank account", "pin", "otp", "cvv"]
    features["personal_info_requests"] = sum(1 for w in personal_info_words if w in full_text)

    # 8. Body length (very short or very long can be suspicious)
    features["body_length"] = len(body)

    # 9. HTML content (phishing often uses heavy HTML)
    features["has_html"] = 1 if re.search(r'<[a-z][\s\S]*>', body) else 0

    # 10. Mismatched display text vs URL
    display_url_mismatches = re.findall(r'<a\s+href=["\']([^"\']+)["\'][^>]*>([^<]+)</a>', body)
    features["url_text_mismatch"] = sum(
        1 for href, text in display_url_mismatches
        if re.search(r'http', text) and href not in text
    )

    return features


# ──────────────────────────────────────────────
# Naive Bayes Classifier (built from scratch)
# ──────────────────────────────────────────────

class PhishingClassifier:
    """
    A simple Gaussian Naive Bayes classifier trained on email features.
    Built from scratch to demonstrate understanding of the algorithm.
    """

    def __init__(self):
        self.class_priors = {}
        self.feature_stats = defaultdict(dict)
        self.classes = []

    def _mean(self, values):
        return sum(values) / len(values)

    def _variance(self, values):
        m = self._mean(values)
        return sum((x - m) ** 2 for x in values) / len(values)

    def train(self, X: list, y: list):
        """Train the classifier on feature dicts and labels."""
        self.classes = list(set(y))
        n_total = len(y)

        for cls in self.classes:
            # Prior probability
            cls_samples = [X[i] for i in range(n_total) if y[i] == cls]
            self.class_priors[cls] = len(cls_samples) / n_total

            # Feature statistics per class
            all_features = cls_samples[0].keys()
            for feature in all_features:
                values = [sample[feature] for sample in cls_samples]
                self.feature_stats[cls][feature] = {
                    "mean": self._mean(values),
                    "var": self._variance(values) + 1e-9  # smoothing
                }

    def _gaussian_prob(self, x, mean, var):
        """Gaussian probability density."""
        exponent = math.exp(-((x - mean) ** 2) / (2 * var))
        return (1 / math.sqrt(2 * math.pi * var)) * exponent

    def predict(self, features: dict) -> tuple:
        """Return predicted class and confidence scores."""
        scores = {}
        for cls in self.classes:
            log_prob = math.log(self.class_priors[cls])
            for feature, value in features.items():
                if feature in self.feature_stats[cls]:
                    mean = self.feature_stats[cls][feature]["mean"]
                    var = self.feature_stats[cls][feature]["var"]
                    prob = self._gaussian_prob(value, mean, var)
                    log_prob += math.log(max(prob, 1e-300))
            scores[cls] = log_prob

        predicted = max(scores, key=scores.get)
        # Convert log probs to rough confidence
        total = sum(math.exp(s) for s in scores.values())
        confidence = {cls: round(math.exp(s) / total * 100, 1) for cls, s in scores.items()}
        return predicted, confidence


# ──────────────────────────────────────────────
# Training Data
# ──────────────────────────────────────────────

TRAINING_EMAILS = [
    # PHISHING examples
    {
        "email": {"subject": "URGENT: Verify your account now!", "body": "Dear customer, your account has been suspended. Click here immediately to verify your account or it will be permanently deleted. http://192.168.1.1/login", "sender": "security@amaz0n-support.xyz"},
        "label": "phishing"
    },
    {
        "email": {"subject": "Action Required: Unusual activity detected", "body": "We have noticed unauthorized access to your bank account. Please click the link below to confirm your identity. http://bit.ly/secure-login123 Failure to verify within 24 hours will result in suspension.", "sender": "noreply@equity-alerts.ru"},
        "label": "phishing"
    },
    {
        "email": {"subject": "Your Password Will Expire!", "body": "Dear valued customer, update your password immediately!! Enter your current password and new password. http://tinyurl.com/updatepass", "sender": "admin@g00gle-security.com"},
        "label": "phishing"
    },
    {
        "email": {"subject": "FINAL NOTICE: Account Locked", "body": "Warning! Your account has been locked due to suspicious activity. Provide your PIN and OTP to restore access now!! Limited time offer to unlock!", "sender": "security@paypa1.net"},
        "label": "phishing"
    },
    {
        "email": {"subject": "Claim your reward NOW!", "body": "Congratulations! You have been selected. Click here urgently to claim your prize. Verify your credit card and bank account details at http://bit.ly/claim-now", "sender": "winner@prize-alert.tk"},
        "label": "phishing"
    },
    {
        "email": {"subject": "Important: Update your KCB account", "body": "Dear account holder, we detected unusual login. Confirm your account details and password immediately. http://kcb-update.xyz/verify", "sender": "alerts@kcb-secure.ru"},
        "label": "phishing"
    },
    # LEGITIMATE examples
    {
        "email": {"subject": "Your monthly statement is ready", "body": "Hi Patricia, your statement for April 2026 is now available. Log in to your account via our official website to view it. Contact us at support@kcbgroup.com if you need help.", "sender": "statements@kcbgroup.com"},
        "label": "legitimate"
    },
    {
        "email": {"subject": "Meeting rescheduled to Thursday", "body": "Hi team, just a quick note that our weekly sync has been moved to Thursday at 2pm. Please update your calendars. Let me know if you have conflicts.", "sender": "manager@company.com"},
        "label": "legitimate"
    },
    {
        "email": {"subject": "Your order has been shipped", "body": "Thank you for your order. Your package has been dispatched and will arrive in 3-5 business days. Track your order using the reference number in your account.", "sender": "orders@jumia.co.ke"},
        "label": "legitimate"
    },
    {
        "email": {"subject": "Welcome to the team!", "body": "Dear Patricia, we are pleased to welcome you to our cybersecurity team. Your onboarding documents are attached. Please review them and reach out to HR with any questions.", "sender": "hr@safaricom.co.ke"},
        "label": "legitimate"
    },
    {
        "email": {"subject": "Seminar: Introduction to Cloud Security", "body": "You are invited to attend our upcoming cloud security seminar on May 20th. Topics include AWS security, zero trust architecture, and incident response best practices.", "sender": "events@isaca.org"},
        "label": "legitimate"
    },
    {
        "email": {"subject": "Invoice #4521 for April services", "body": "Please find attached your invoice for services rendered in April 2026. Payment is due within 30 days. Thank you for your continued business.", "sender": "billing@techservices.co.ke"},
        "label": "legitimate"
    },
]


def build_training_data():
    X, y = [], []
    for item in TRAINING_EMAILS:
        X.append(extract_features(item["email"]))
        y.append(item["label"])
    return X, y


# ──────────────────────────────────────────────
# Analysis Report
# ──────────────────────────────────────────────

def generate_report(email: dict, features: dict, prediction: str, confidence: dict) -> str:
    """Generate a human-readable analysis report."""
    verdict_icon = "🚨 PHISHING" if prediction == "phishing" else "✅ LEGITIMATE"
    conf = confidence.get(prediction, 0)

    red_flags = []
    if features["urgency_count"] > 1:
        red_flags.append(f"  ⚠  Contains {features['urgency_count']} urgency trigger words")
    if features["suspicious_phrases"] > 0:
        red_flags.append(f"  ⚠  Contains {features['suspicious_phrases']} suspicious phrases")
    if features["suspicious_urls"] > 0:
        red_flags.append(f"  ⚠  Found {features['suspicious_urls']} suspicious URL(s) (IP-based, shortened, or typosquatted)")
    if features["personal_info_requests"] > 0:
        red_flags.append(f"  ⚠  Requests sensitive info (password, PIN, card details)")
    if features["trusted_sender"] == 0 and features["url_count"] > 0:
        red_flags.append("  ⚠  Sender domain is not in trusted list")
    if features["exclamation_count"] > 2:
        red_flags.append(f"  ⚠  Excessive exclamation marks ({features['exclamation_count']})")
    if features["caps_ratio"] > 0.15:
        red_flags.append(f"  ⚠  High proportion of capital letters ({features['caps_ratio']*100:.0f}%)")

    report = f"""
╔══════════════════════════════════════════════════════════╗
║           PHISHING EMAIL DETECTOR — ANALYSIS REPORT      ║
╚══════════════════════════════════════════════════════════╝

  Subject : {email.get('subject', 'N/A')}
  Sender  : {email.get('sender', 'N/A')}

  VERDICT : {verdict_icon}
  Confidence: {conf}%

──────────────────────────────────────────────────────────
  FEATURE BREAKDOWN
──────────────────────────────────────────────────────────
  Urgency words found   : {features['urgency_count']}
  Suspicious phrases    : {features['suspicious_phrases']}
  URLs detected         : {features['url_count']}
  Suspicious URLs       : {features['suspicious_urls']}
  Personal info requests: {features['personal_info_requests']}
  Trusted sender domain : {'Yes' if features['trusted_sender'] else 'No'}
  Exclamation marks     : {features['exclamation_count']}
  Capital letter ratio  : {features['caps_ratio']*100:.1f}%
  Contains HTML         : {'Yes' if features['has_html'] else 'No'}

──────────────────────────────────────────────────────────
  RED FLAGS
──────────────────────────────────────────────────────────
"""
    if red_flags:
        report += "\n".join(red_flags)
    else:
        report += "  None detected."

    report += "\n\n══════════════════════════════════════════════════════════\n"
    return report


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def analyze_email(email: dict, classifier: PhishingClassifier) -> str:
    features = extract_features(email)
    prediction, confidence = classifier.predict(features)
    return generate_report(email, features, prediction, confidence)


def main():
    print("\n  Training classifier on sample data...")
    X, y = build_training_data()
    classifier = PhishingClassifier()
    classifier.train(X, y)
    print("  Classifier ready.\n")

    # Test emails
    test_emails = [
        {
            "subject": "URGENT: Your Safaricom account will be suspended!",
            "body": "Dear customer, we detected unusual activity on your account. Click here immediately to verify your identity or your account will be locked. http://bit.ly/safaricom-verify Enter your PIN and OTP to confirm.",
            "sender": "security@safar1com-alert.xyz"
        },
        {
            "subject": "Team lunch this Friday",
            "body": "Hey everyone, just a reminder that we have team lunch this Friday at 1pm at the usual spot. Please RSVP by Thursday. Looking forward to seeing everyone!",
            "sender": "teamlead@company.co.ke"
        },
        {
            "subject": "Action Required: Verify your M-Pesa account",
            "body": "Warning! Your M-Pesa account has been flagged. Provide your password and credit card details at http://192.168.10.5/mpesa-verify to avoid permanent suspension!!",
            "sender": "mpesa-support@safar1com.ru"
        },
        {
            "subject": "Your USIU exam results are available",
            "body": "Dear student, your results for the Spring 2026 semester have been uploaded to the student portal. Log in at usiu.ac.ke to view your grades. Contact the registrar for any queries.",
            "sender": "registrar@usiu.ac.ke"
        }
    ]

    for email in test_emails:
        print(analyze_email(email, classifier))

    # Interactive mode
    print("\n  ── INTERACTIVE MODE ──")
    print("  Paste an email to analyze (or press Enter to skip)\n")
    subject = input("  Email subject: ").strip()
    if subject:
        body = input("  Email body: ").strip()
        sender = input("  Sender address: ").strip()
        email = {"subject": subject, "body": body, "sender": sender}
        print(analyze_email(email, classifier))


if __name__ == "__main__":
    main()
