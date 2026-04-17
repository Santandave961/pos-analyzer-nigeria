import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import random

st.set_page_config(page_title="POS Analyzer Nigeria", layout="centered")


# ── Synthetic Transaction Data ────────────────────────────────────────────────
@st.cache_resource
def generate_data():
    random.seed(42)
    np.random.seed(42)

    states = ["Lagos","Abuja","Kano","Rivers","Oyo","Kaduna","Enugu","Delta","Anambra","Ogun"]
    categories = ["Food & Beverage","Transport","Retail","Healthcare","Entertainment",
                  "Utilities","Education","Fuel","Fashion","Electronics"]
    banks = ["GTBank","Access Bank","First Bank","Zenith Bank","UBA","Kuda","Opay","Moniepoint"]

    n = 500
    dates = [datetime(2024,1,1) + timedelta(days=random.randint(0,365)) for _ in range(n)]

    # Normal transactions
    amounts = np.random.lognormal(mean=9.5, sigma=1.2, size=n)  # NGN amounts

    # Inject anomalies (5%)
    anomaly_idx = random.sample(range(n), 25)
    for i in anomaly_idx:
        amounts[i] = amounts[i] * random.uniform(8, 20)  # suspiciously large

    data = {
        "transaction_id": [f"TXN{str(i).zfill(5)}" for i in range(n)],
        "date":           dates,
        "amount_ngn":     amounts.round(2),
        "state":          [random.choice(states) for _ in range(n)],
        "category":       [random.choice(categories) for _ in range(n)],
        "bank":           [random.choice(banks) for _ in range(n)],
        "hour":           [random.randint(0, 23) for _ in range(n)],
        "is_weekend":     [1 if d.weekday() >= 5 else 0 for d in dates],
        "true_anomaly":   [1 if i in anomaly_idx else 0 for i in range(n)],
    }

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek

    # Encode
    le_state = LabelEncoder()
    le_cat   = LabelEncoder()
    le_bank  = LabelEncoder()
    df["state_enc"] = le_state.fit_transform(df["state"])
    df["cat_enc"]   = le_cat.fit_transform(df["category"])
    df["bank_enc"]  = le_bank.fit_transform(df["bank"])

    # Train Isolation Forest
    feat_cols = ["amount_ngn","hour","is_weekend","state_enc","cat_enc","bank_enc","month","day_of_week"]
    X = df[feat_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    df["anomaly_score"] = iso.fit_predict(X_scaled)
    df["is_anomaly"]    = (df["anomaly_score"] == -1).astype(int)
    df["anomaly_label"] = df["is_anomaly"].map({0: "Normal", 1: "Anomaly"})

    return df, feat_cols, iso, scaler, le_state, le_cat, le_bank


df, feat_cols, iso, scaler, le_state, le_cat, le_bank, = generate_data()


# ── Header ────────────────────────────────────────────────────────────────────
st.title("POS Transaction Analyzer")
st.caption("Nigerian POS Transaction Dashboard + ML Anomaly Detection")
st.markdown("Analyze spending patterns across Nigerian states and merchant categories, with Isolation Forest anomaly detection.")
st.divider()

# ── KPI Metrics ───────────────────────────────────────────────────────────────
st.subheader("Overview")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Transactions", f"{len(df):,}")
k2.metric("Total Volume",       f"N{df['amount_ngn'].sum()/1e6:.1f}M")
k3.metric("Avg Transaction",    f"N{df['amount_ngn'].mean():,.0f}")
k4.metric("Anomalies Detected", f"{df['is_anomaly'].sum()} ({df['is_anomaly'].mean()*100:.1f}%)")
st.divider()

# ── Filters ───────────────────────────────────────────────────────────────────
st.subheader("Filters")
col1, col2, col3 = st.columns(3)
with col1:
    states_all  = ["All"] + sorted(df["state"].unique().tolist())
    sel_state   = st.selectbox("State", states_all)
with col2:
    cats_all    = ["All"] + sorted(df["category"].unique().tolist())
    sel_cat     = st.selectbox("Category", cats_all)
with col3:
    banks_all   = ["All"] + sorted(df["bank"].unique().tolist())
    sel_bank    = st.selectbox("Bank", banks_all)

# Apply filters
fdf = df.copy()
if sel_state != "All": fdf = fdf[fdf["state"]    == sel_state]
if sel_cat   != "All": fdf = fdf[fdf["category"] == sel_cat]
if sel_bank  != "All": fdf = fdf[fdf["bank"]     == sel_bank]

st.caption(f"Showing {len(fdf):,} transactions after filters")
st.divider()

# ── Charts ────────────────────────────────────────────────────────────────────
st.subheader("Spending Analysis")
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "By State", "By Category", "By Bank", "Time Trends", "Anomalies"
])

with tab1:
    st.markdown("#### Transaction Volume by State")
    state_vol = fdf.groupby("state")["amount_ngn"].sum().sort_values(ascending=True) / 1e6
    fig, ax   = plt.subplots(figsize=(7, 4))
    colors    = ["#e74c3c" if s == state_vol.idxmax() else "#3498db" for s in state_vol.index]
    bars      = ax.barh(state_vol.index, state_vol.values, color=colors, height=0.6)
    ax.set_xlabel("Total Volume (NGN Millions)")
    ax.set_title("POS Transaction Volume by State")
    for bar, val in zip(bars, state_vol.values):
        ax.text(val + 0.1, bar.get_y() + bar.get_height()/2,
                f"N{val:.1f}M", va="center", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("#### Transaction Count by State")
    state_cnt = fdf["state"].value_counts().sort_values(ascending=True)
    fig, ax   = plt.subplots(figsize=(7, 4))
    ax.barh(state_cnt.index, state_cnt.values, color="#2ecc71", height=0.6)
    ax.set_xlabel("Number of Transactions")
    ax.set_title("POS Transaction Count by State")
    for i, (idx, val) in enumerate(state_cnt.items()):
        ax.text(val + 0.5, i, str(val), va="center", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with tab2:
    st.markdown("#### Spending by Merchant Category")
    cat_vol = fdf.groupby("category")["amount_ngn"].sum().sort_values(ascending=True) / 1e6
    fig, ax = plt.subplots(figsize=(7, 4))
    colors  = ["#e74c3c" if c == cat_vol.idxmax() else "#9b59b6" for c in cat_vol.index]
    bars    = ax.barh(cat_vol.index, cat_vol.values, color=colors, height=0.6)
    ax.set_xlabel("Total Volume (NGN Millions)")
    ax.set_title("Spending by Merchant Category")
    for bar, val in zip(bars, cat_vol.values):
        ax.text(val + 0.1, bar.get_y() + bar.get_height()/2,
                f"N{val:.1f}M", va="center", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("#### Average Transaction by Category")
    cat_avg = fdf.groupby("category")["amount_ngn"].mean().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(cat_avg.index, cat_avg.values/1000, color="#f39c12", height=0.6)
    ax.set_xlabel("Avg Transaction (NGN Thousands)")
    ax.set_title("Average Transaction Size by Category")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with tab3:
    st.markdown("#### Transaction Volume by Bank")
    bank_vol = fdf.groupby("bank")["amount_ngn"].sum().sort_values(ascending=True) / 1e6
    fig, ax  = plt.subplots(figsize=(7, 4))
    colors   = ["#e74c3c" if b == bank_vol.idxmax() else "#1abc9c" for b in bank_vol.index]
    bars     = ax.barh(bank_vol.index, bank_vol.values, color=colors, height=0.6)
    ax.set_xlabel("Total Volume (NGN Millions)")
    ax.set_title("POS Volume by Bank")
    for bar, val in zip(bars, bank_vol.values):
        ax.text(val + 0.1, bar.get_y() + bar.get_height()/2,
                f"N{val:.1f}M", va="center", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with tab4:
    st.markdown("#### Monthly Transaction Trend")
    monthly = fdf.groupby("month")["amount_ngn"].sum() / 1e6
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(monthly.index, monthly.values, marker="o", color="#3498db",
            linewidth=2.5, markersize=7)
    ax.fill_between(monthly.index, monthly.values, alpha=0.15, color="#3498db")
    ax.set_xticks(monthly.index)
    ax.set_xticklabels([month_names[m-1] for m in monthly.index], rotation=45)
    ax.set_ylabel("Volume (NGN Millions)")
    ax.set_title("Monthly POS Transaction Volume")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("#### Transactions by Hour of Day")
    hourly = fdf.groupby("hour")["amount_ngn"].count()
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(hourly.index, hourly.values, color="#e67e22", width=0.8)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Number of Transactions")
    ax.set_title("When Do POS Transactions Happen?")
    ax.axvline(12, color="red", linestyle="--", linewidth=1, label="Noon")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("#### Weekday vs Weekend")
    wk = fdf.groupby("is_weekend")["amount_ngn"].agg(["sum","count"])
    wk.index = ["Weekday","Weekend"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))
    ax1.bar(wk.index, wk["sum"]/1e6, color=["#3498db","#e74c3c"])
    ax1.set_ylabel("Volume (NGN Millions)")
    ax1.set_title("Volume: Weekday vs Weekend")
    ax2.bar(wk.index, wk["count"], color=["#2ecc71","#f39c12"])
    ax2.set_ylabel("Count")
    ax2.set_title("Count: Weekday vs Weekend")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with tab5:
    st.markdown("#### Anomaly Detection — Isolation Forest")
    st.markdown(f"**{fdf['is_anomaly'].sum()} anomalous transactions** detected out of {len(fdf):,}")

    # Scatter plot
    fig, ax = plt.subplots(figsize=(7, 4))
    normal  = fdf[fdf["is_anomaly"] == 0]
    anomaly = fdf[fdf["is_anomaly"] == 1]
    ax.scatter(normal["date"],  normal["amount_ngn"]/1000,
               color="#3498db", alpha=0.4, s=20, label="Normal")
    ax.scatter(anomaly["date"], anomaly["amount_ngn"]/1000,
               color="#e74c3c", alpha=0.9, s=60, marker="x", label="Anomaly", linewidths=2)
    ax.set_xlabel("Date")
    ax.set_ylabel("Amount (NGN Thousands)")
    ax.set_title("Anomalous Transactions Over Time")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("#### Anomalies by State")
    anom_state = fdf[fdf["is_anomaly"]==1]["state"].value_counts()
    fig, ax    = plt.subplots(figsize=(7, 3.5))
    ax.bar(anom_state.index, anom_state.values, color="#e74c3c", width=0.6)
    ax.set_xlabel("State")
    ax.set_ylabel("Anomaly Count")
    ax.set_title("Where Are Anomalies Occurring?")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("#### Anomalies by Category")
    anom_cat = fdf[fdf["is_anomaly"]==1]["category"].value_counts()
    fig, ax  = plt.subplots(figsize=(7, 3.5))
    ax.barh(anom_cat.index, anom_cat.values, color="#e74c3c", height=0.6)
    ax.set_xlabel("Anomaly Count")
    ax.set_title("Which Categories Have Most Anomalies?")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("#### Flagged Transactions")
    anomaly_table = fdf[fdf["is_anomaly"]==1][
        ["transaction_id","date","state","category","bank","amount_ngn"]
    ].sort_values("amount_ngn", ascending=False).head(20)
    anomaly_table["amount_ngn"] = anomaly_table["amount_ngn"].apply(lambda x: f"N{x:,.0f}")
    st.dataframe(anomaly_table.reset_index(drop=True), use_container_width=True)

st.divider()

# ── Single Transaction Check ──────────────────────────────────────────────────
st.subheader("Check a Single Transaction")
st.markdown("Input a transaction to check if it is flagged as anomalous.")

tc1, tc2 = st.columns(2)
with tc1:
    t_amount   = st.number_input("Amount (NGN)", min_value=100, max_value=10000000, value=5000)
    t_hour     = st.slider("Hour of Day", 0, 23, 14)
    t_weekend  = st.selectbox("Day Type", ["Weekday", "Weekend"])
    t_state    = st.selectbox("State", sorted(df["state"].unique().tolist()))
with tc2:
    t_category = st.selectbox("Merchant Category", sorted(df["category"].unique().tolist()))
    t_bank     = st.selectbox("Bank", sorted(df["bank"].unique().tolist()))
    t_month    = st.slider("Month", 1, 12, 6)
    t_dow      = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 2)

if st.button("Check Transaction", use_container_width=True):
    t_wk  = 1 if t_weekend == "Weekend" else 0
    t_s   = le_state.transform([t_state])[0]
    t_c   = le_cat.transform([t_category])[0]
    t_b   = le_bank.transform([t_bank])[0]

    X_t   = np.array([[t_amount, t_hour, t_wk, t_s, t_c, t_b, t_month, t_dow]])
    X_ts  = scaler.transform(X_t)
    pred  = iso.predict(X_ts)[0]
    score = iso.decision_function(X_ts)[0]

    if pred == -1:
        st.error("ANOMALY DETECTED - This transaction looks suspicious!")
        st.metric("Anomaly Score", f"{score:.4f}", "Lower = more anomalous")
    else:
        st.success("NORMAL - This transaction looks legitimate.")
        st.metric("Anomaly Score", f"{score:.4f}", "Higher = more normal")

st.divider()

# ── Technical Details ─────────────────────────────────────────────────────────
with st.expander("Technical Details"):
    st.markdown(
        "- **Algorithm:** Isolation Forest\n"
        "- **Estimators:** 200 trees\n"
        "- **Contamination:** 5% (expected anomaly rate)\n"
        "- **Scaler:** StandardScaler\n"
        "- **Features:** Amount, Hour, Weekend flag, State, Category, Bank, Month, Day of Week\n"
        "- **Dataset:** " + str(len(df)) + " synthetic Nigerian POS transactions\n"
        "- **Anomalies injected:** 25 (5%) with amounts 8-20x normal\n"
        "- **Use case:** Real-time POS fraud and anomaly flagging"
    )