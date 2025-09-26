## Antifraud Challenge

This repository contains my solution for the **recruitment challenge**.  
The task is to analyze transactional data, detect suspicious behaviors, and propose an anti-fraud strategy that balances security, customer experience, and system performance.

---

### Problem Statement
I received a dataset of card-not-present (CNP, likely cellphone) transactions with the following fields:

```json
{
  "transaction_id": 2342357,
  "merchant_id": 29744,
  "user_id": 97051,
  "card_number": "434505******9116",
  "transaction_date": "2019-11-31T23:16:32.812632",
  "transaction_amount": 373,
  "device_id": 285475,
  "has_cbk": false
}

The goal is to:
Identify suspicious patterns in the data.
Propose relevant external data sources that would improve fraud detection.
Recommend fraud prevention measures to reduce chargebacks.
Design a conceptual/technical solution (rules, ML model, or hybrid).
Present findings with clear reasoning, tables, and charts.

Tech Stack
Python 3.10+
pandas
 – data wrangling
numpy
 – numerical computations
matplotlib
 – visualization
luhn
 – card number validity checks
flask
 – minimal API prototype

### Data Sanity Checks

The first step was cleaning and validating the dataset:
Null handling: ~26% of device_id values missing. Flagged as potential fraud indicator.
Transaction dates: parsed and validated.
Uniqueness: ensured transaction_id is unique.
Amounts: checked for non-positive and extreme outliers.
Card format: verified consistent masking pattern (BIN + last 4).
Chargeback prevalence: baseline fraud rate calculated from has_cbk.

Fraud Indicators Considered
Multiple rapid attempts by the same user_id or device_id.
High-value transactions above dynamic thresholds.
Devices shared across many users (possible mule accounts).
Missing device_id combined with high transaction amounts.
Users with past chargebacks (has_cbk=True) → higher risk going forward.

Solution Design
Hybrid approach (rules + scoring model):
Rules engine:
Reject too many attempts in short period.
Reject transactions above amount threshold per time window.
Reject if user has previous chargeback.

Scoring model (prototype):
Logistic Regression trained on features: transaction frequency, device sharing, transaction amount percentile, past CBK history.
Produces risk score → approve / manual review / reject.
Architecture sketch:
Data ingestion (CSV / API) →
Sanity checks + feature engineering →
Rule engine evaluation →
ML model scoring →
Decision output (approve, manual_review, reject).

Results & Insights
Fraudulent transactions (has_cbk=True) cluster around:
Higher transaction amounts.
Missing or reused device_ids.
Certain high-risk merchants.
Rules reduced chargeback exposure by X% (placeholder for your actual result).

How to Run
Clone this repository:
git clone https://github.com/<your-username>/cloudwalk-antifraud.git
cd cloudwalk-antifraud

Install dependencies:
pip install -r requirements.txt

Run exploratory analysis:
python analysis/sanity_checks.py

Run fraud detection prototype:

python antifraud/main.py

Next Steps
Expand model with external signals (IP, geolocation, device fingerprint).
Add real-time streaming detection (Kafka + Flask API).
Implement monitoring dashboards (Looker / Power BI / Tableau).