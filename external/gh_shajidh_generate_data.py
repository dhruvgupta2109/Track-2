import argparse
import numpy as np
import pandas as pd
from .config import DATA_FILE, DATA_DIR, RANDOM_STATE

MERCHANTS = ["grocery","electronics","restaurant","travel","fuel",
             "online_shopping","pharmacy","gaming","jewelry","cash_withdrawal"]

def generate_transactions(n_rows=12000, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    amount = np.round(np.clip(rng.lognormal(3.7, 1.05, n_rows), 5, 5000), 2)
    hour = rng.integers(0, 24, n_rows)
    age = rng.integers(18, 76, n_rows)
    account_age = np.clip(rng.gamma(4, 260, n_rows).astype(int), 10, 5000)
    tx24 = rng.poisson(2.4, n_rows)
    avg30 = np.round(np.clip(rng.lognormal(3.4, .65, n_rows), 10, 2500), 2)
    distance = np.round(np.clip(rng.exponential(15, n_rows), 0, 500), 2)
    device_trust = np.round(rng.beta(7, 2, n_rows) * 100, 2)
    failed = np.clip(rng.poisson(.5, n_rows), 0, 8)
    intl = rng.binomial(1, .12, n_rows)
    card = rng.binomial(1, .58, n_rows)
    new_device = rng.binomial(1, .10, n_rows)
    merchant = rng.choice(MERCHANTS, n_rows, p=[.16,.11,.15,.08,.12,.16,.07,.05,.03,.07])

    # Strong but noisy synthetic risk function. The signal is intentionally
    # learnable so the demo model produces meaningful evaluation metrics.
    logit = (
        -4.6
        + 0.0010 * amount
        + 1.35 * (amount > (avg30 * 2.5)).astype(float)
        + 1.20 * intl
        + 1.35 * new_device
        + 1.10 * (device_trust < 45)
        + 0.28 * failed
        + 0.22 * tx24
        + 1.30 * ((hour <= 4) | (hour >= 23)).astype(float)
        + 1.00 * (distance > 80)
        + 0.55 * (card == 0)
        + 0.80 * np.isin(merchant, ["gaming", "jewelry", "cash_withdrawal"])
        + 0.75 * (account_age < 60)
        + rng.normal(0, 0.25, n_rows)
    )
    probability = 1 / (1 + np.exp(-logit))
    fraud = rng.binomial(1, probability)

    # Keep the demo dataset in a useful fraud-rate range.
    if fraud.mean() < 0.03 or fraud.mean() > 0.15:
        fraud = (probability >= np.quantile(probability, 0.93)).astype(int)

    return pd.DataFrame({
        "transaction_id": [f"TXN-{i:08d}" for i in range(1, n_rows+1)],
        "amount": amount, "transaction_hour": hour, "merchant_category": merchant,
        "customer_age": age, "account_age_days": account_age,
        "transactions_last_24h": tx24, "avg_amount_30d": avg30,
        "distance_from_home_km": distance, "device_trust_score": device_trust,
        "failed_login_attempts": failed, "is_international": intl,
        "card_present": card, "is_new_device": new_device, "is_fraud": fraud
    })

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=12000)
    args = parser.parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = generate_transactions(args.rows)
    df.to_csv(DATA_FILE, index=False)
    print(f"Created {DATA_FILE} with {len(df):,} rows; fraud rate={df.is_fraud.mean():.2%}")
