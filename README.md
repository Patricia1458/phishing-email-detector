
# Phishing Email Detector

A Python-based machine learning tool that analyzes emails and classifies them as **phishing** or **legitimate** using a Naive Bayes classifier built from scratch.

---

## Background

Phishing remains one of the most common attack vectors in cybersecurity. This project was built to demonstrate how machine learning can be applied to detect phishing emails by extracting behavioral and linguistic features from email content.

Inspired by real-world phishing analysis experience gained through the **Mastercard Cybersecurity Virtual Experience Program** on Forage.

---

## How It Works

The detector works in 3 stages:

### 1. Feature Extraction
Each email is analyzed for 10+ features including:
- **Urgency trigger words** — "act now", "account suspended", "verify immediately"
- **Suspicious phrases** — "dear customer", "click the link below", "we have noticed"
- **URL analysis** — IP-based URLs, shortened links (bit.ly, tinyurl), typosquatted domains (paypa1, arnazon)
- **Sender domain trust** — checks against a list of known legitimate domains
- **Personal info requests** — password, PIN, OTP, CVV, credit card
- **Text patterns** — excessive caps, exclamation marks, HTML content

### 2. Classification (Naive Bayes)
A **Gaussian Naive Bayes** classifier is trained on labeled email samples. The algorithm:
- Learns the statistical distribution of each feature per class
- Uses Bayes' theorem to calculate the probability of phishing vs legitimate
- Returns a prediction with a confidence percentage

The classifier is **built from scratch** (no sklearn) to demonstrate understanding of the underlying math.

### 3. Analysis Report
The tool generates a structured report showing:
- Final verdict (Phishing / Legitimate)
- Confidence score
- Feature breakdown
- Specific red flags detected

---

## Usage

### Requirements
```bash
Python 3.8+
No external libraries required
```

### Run the detector
```bash
python detector.py
```

This will:
1. Train the classifier on built-in sample data
2. Analyze 4 test emails and print reports
3. Prompt you to enter your own email for analysis

### Example Output
```
╔══════════════════════════════════════════════════════════╗
║           PHISHING EMAIL DETECTOR — ANALYSIS REPORT      ║
╚══════════════════════════════════════════════════════════╝

  Subject : URGENT: Your Safaricom account will be suspended!
  Sender  : security@safar1com-alert.xyz

  VERDICT : PHISHING
  Confidence: 94.3%

  RED FLAGS
  ⚠  Contains 3 urgency trigger words
  ⚠  Found 1 suspicious URL(s) (shortened link)
  ⚠  Requests sensitive info (PIN, OTP)
  ⚠  Sender domain is not in trusted list
```

---

## Project Structure

```
phishing-detector/
│
├── detector.py        # Main script — feature extraction, classifier, reports
└── README.md          # Project documentation
```

---

## Key Concepts Demonstrated

| Concept | Implementation |
|---|---|
| Feature Engineering | 10+ handcrafted features from email metadata and content |
| Naive Bayes | Gaussian NB built from scratch using probability math |
| Regular Expressions | URL extraction, domain parsing, HTML detection |
| Security Awareness | Real-world phishing indicators from industry practice |
| Clean Code | Modular functions, docstrings, clear separation of concerns |

---

##  Disclaimer

This tool is built for **educational purposes**. It is not a replacement for enterprise-grade email security solutions like spam filters or SEGs. Always use official security tools in production environments.

---

## Author

**Patricia Naamala**  
Cybersecurity Student — USIU-Africa  
[LinkedIn](https://linkedin.com/in/patricianamala) | [GitHub](https://github.com/Patricia1458)
