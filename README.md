# POS Analyzer Nigeria - POS Transaction Dashboard + ML Anomaly Detection

A data science web application that analyzes Nigerian Point of Sale (POS) transaction patterns across states and merchant categories, with Isolation Forest machine learning anomaly detection.

Built with Python and Streamlit, deployed on Streamlit Community Cloud.

---

## Live Demo

[Click here to view the app]https://pos-analyzer-nigeria-7esqtzaid3vrd3yneznj83.streamlit.app/

---

## Overview

This project builds an interactive analytics dashboard that gives visibility into Nigerian POS transaction behaviour across 10 states, 10 merchant categories, and 8 banks — while using an unsupervised machine learning model to automatically flag suspicious transactions in real time.

---

## Model Performance

| Metric | Value |
|--------|-------|
| Algorithm | Isolation Forest |
| Estimators | 200 trees |
| Contamination | 5% |
| Scaler | StandardScaler |
| Dataset | 500 synthetic Nigerian POS transactions |
| Anomalies Injected | 25 (5%) with amounts 8-20x normal |

---

## Features

**Dashboard:**
- Total transaction volume, count, average, and anomaly rate KPIs
- Filter by State, Category, and Bank
- Transaction volume and count by Nigerian state
- Spending breakdown by merchant category
- Volume comparison by bank
- Monthly transaction trend
- Hour of day activity chart
- Weekday vs Weekend analysis

**ML Anomaly Detection:**
- Isolation Forest trained on 500 transactions
- Anomalous transactions scatter plot over time
- Anomalies by state and category
- Flagged transactions table
- Single transaction checker — input any transaction and get instant Normal or Anomaly verdict

---

## States Covered

Lagos, Abuja, Kano, Rivers, Oyo, Kaduna, Enugu, Delta, Anambra, Ogun

---

## Merchant Categories

Food and Beverage, Transport, Retail, Healthcare, Entertainment, Utilities, Education, Fuel, Fashion, Electronics

---

## Banks Covered

GTBank, Access Bank, First Bank, Zenith Bank, UBA, Kuda, Opay, Moniepoint

---

## Tech Stack

- **Language:** Python 3
- **Framework:** Streamlit
- **ML Library:** scikit-learn
- **Algorithm:** Isolation Forest (unsupervised anomaly detection)
- **Scaler:** StandardScaler
- **Data Processing:** pandas, NumPy
- **Visualisation:** Matplotlib

---

## Project Structure

```
pos-analyzer-nigeria/
    app.py              # Main Streamlit application
    requirements.txt    # Python dependencies
    README.md           # Project documentation
```

---

## How to Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/Santandave961/pos-analyzer-nigeria.git
cd pos-analyzer-nigeria
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## How It Works

1. **Data** - 500 synthetic Nigerian POS transactions with 25 injected anomalies (amounts 8-20x normal)
2. **Features** - Amount, hour of day, weekend flag, state, category, bank, month, day of week
3. **Preprocessing** - LabelEncoder for categoricals, StandardScaler for normalization
4. **Model** - Isolation Forest with 5% contamination rate detects statistical outliers
5. **Output** - Anomaly flag (Normal or Anomaly) with decision score for each transaction

---

## Key Insights

- **Lagos** consistently records the highest POS transaction volume
- **Food and Beverage** and **Retail** are the top spending categories
- Peak transaction hours are between **11am and 3pm**
- Weekday transactions outnumber weekend transactions significantly
- Anomalies are spread across all states and categories with no clear concentration

---

## Use Cases

- Fintech companies monitoring POS terminal activity for fraud
- Banks tracking merchant category spending patterns
- CBN and regulators monitoring state-level transaction volumes
- Data analysts building financial crime detection pipelines

---

## Author

**Okparaji Wisdom**
Data Science Student | Fintech Portfolio Builder

- GitHub: [@Santandave961](https://github.com/Santandave961)
- LinkedIn: [Connect with me](https://linkedin.com)

---

## License

MIT License - feel free to use and modify this project.
