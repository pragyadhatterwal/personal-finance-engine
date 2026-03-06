# extractors/sms_parser.py

import re
from datetime import datetime
from typing import Dict, Optional

# Common regex patterns for Indian banking/UPI SMS formats
AMOUNT_PATTERN = r"(?:(?:INR|Rs\.?|₹)\s*)(?P<amount>\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)"
DATE_FALLBACK_FMT = "%Y-%m-%d"

DEBIT_KEYWORDS = [
    r"debited", r"spent", r"withdrawn", r"purchase", r"paid", r"upi txn", r"imps", r"neft",
    r"debit of", r"dr", r"payment"
]
CREDIT_KEYWORDS = [
    r"credited", r"received", r"refund", r"credit of", r"cr", r"salary", r"cashback"
]

# Merchant / Payee extraction heuristics for UPI/merchant lines
MERCHANT_PATTERNS = [
    r"at\s+(?P<merchant>[A-Za-z0-9& ._-]{2,40})",             # "debited at Swiggy"
    r"to\s+(?P<merchant>[A-Za-z0-9& ._-]{2,40})",             # "paid to Amazon"
    r"by\s+(?P<merchant>[A-Za-z0-9& ._-]{2,40})",             # "credited by ABC Ltd"
    r"UPI:(?P<merchant>[a-zA-Z0-9.\-_@]{5,100})",             # "UPI: abcd@okhdfcbank"
    r"via\s+(?P<merchant>[A-Za-z0-9& ._-]{2,40})"
]

BALANCE_PATTERNS = [
    r"av(?:ail)?\.?\s*bal(?:ance)?:?\s*(?:INR|Rs\.?|₹)?\s*(?P<balance>[\d,]+(?:\.\d{1,2})?)",
    r"bal(?:ance)?:\s*(?:INR|Rs\.?|₹)?\s*(?P<balance>[\d,]+(?:\.\d{1,2})?)"
]

def _clean_amount(text_amt: str) -> float:
    return float(text_amt.replace(",", ""))

def _infer_type(text: str) -> str:
    t = text.lower()
    if any(re.search(k, t) for k in CREDIT_KEYWORDS):
        return "credit"
    if any(re.search(k, t) for k in DEBIT_KEYWORDS):
        return "debit"
    # Fallback: heuristics
    return "debit" if "-" in t else "credit"

def _extract_merchant(text: str) -> Optional[str]:
    for pat in MERCHANT_PATTERNS:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m and m.groupdict().get("merchant"):
            merchant = m.group("merchant").strip(" .:-")
            # Normalize UPI handles like name@bank -> name
            if "@" in merchant and len(merchant) > 4:
                return merchant.split("@")[0]
            return merchant
    return None

def _extract_balance(text: str) -> Optional[float]:
    for pat in BALANCE_PATTERNS:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m and m.groupdict().get("balance"):
            try:
                return _clean_amount(m.group("balance"))
            except:
                continue
    return None

def parse_sms(text: str, received_at: Optional[datetime] = None) -> Optional[Dict]:
    """
    Parse a raw SMS string and return a normalized transaction dict.
    Returns None if no amount is found (likely non-transactional SMS).
    """
    amt_match = re.search(AMOUNT_PATTERN, text, flags=re.IGNORECASE)
    if not amt_match:
        return None

    amount = _clean_amount(amt_match.group("amount"))
    tx_type = _infer_type(text)
    merchant = _extract_merchant(text) or "Unknown"
    balance = _extract_balance(text)

    # Prefer provided timestamp else fallback to now
    ts = received_at or datetime.now()

    return {
        "date": ts.strftime(DATE_FALLBACK_FMT),
        "datetime": ts.isoformat(timespec="seconds"),
        "merchant": merchant,
        "amount": amount,
        "type": tx_type,
        "source": "SMS",
        "raw": text,
        "balance": balance
    }

if __name__ == "__main__":
    samples = [
        "INR 462.50 has been debited from your HDFC A/C via UPI: swiggy@okhdfcbank. Avl bal: INR 23,540.90.",
        "Rs. 5,000 credited to your SBI account by Amazon Refund. Avail bal Rs 35,000.",
        "₹1,299.00 spent at Flipkart. Avl bal: 12,340.50"
    ]
    for s in samples:
        print(parse_sms(s))
