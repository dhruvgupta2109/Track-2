import numpy as np
import pandas as pd

CAT_COLS = ["merchant_category", "country", "transaction_channel"]
RAW_NUM_COLS = ["transaction_amount", "transaction_hour", "transactions_last_24h", "spend_last_24h",
                "account_age", "new_device", "transactions_last_1h"]


def make_features(df):
    """Row-wise feature engineering, shared verbatim by train.ipynb and predict.ipynb.

    Uses only the columns of the row itself: never `id`, never statistics over the input
    file, never row order. Missing inputs stay NaN (the model pipeline handles them).
    """
    df = df.copy()
    df = df.drop(columns=[c for c in ("id", "fraud") if c in df.columns])
    for c in CAT_COLS:
        df[c] = df[c].astype(object).where(df[c].notna(), np.nan)
    for c in RAW_NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)

    amt = df["transaction_amount"]
    n24 = df["transactions_last_24h"]
    s24 = df["spend_last_24h"]
    n1 = df["transactions_last_1h"]
    age = df["account_age"]
    hour = df["transaction_hour"]

    df["log_amount"] = np.log1p(amt)
    df["log_spend_24h"] = np.log1p(s24)
    df["log_account_age"] = np.log1p(age)
    df["avg_spend_24h"] = s24 / n24                       # n24 >= 1 in the data
    df["amt_to_avg_24h"] = amt / (df["avg_spend_24h"] + 1.0)
    df["amt_to_spend_24h"] = amt / (s24 + 1.0)
    df["velocity_1h_24h"] = n1 / n24
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["night"] = hour.isin([23, 0, 1, 2, 3, 4]).astype(float)
    df["new_device_x_log_amount"] = df["new_device"] * df["log_amount"]
    df["new_device_x_new_account"] = _flag(age < 180, age) * df["new_device"]

    # Risk flags at the steps seen in the EDA (fixed constants, not learned from the input file)
    df["high_amount"] = _flag(amt > 500, amt)
    df["young_account"] = _flag(age <= 120, age)
    df["account_lt_1y"] = _flag(age <= 365, age)
    df["burst_1h"] = _flag(n1 >= 3, n1)
    df["burst_24h"] = _flag(n24 >= 11, n24)
    df["high_spend_24h"] = _flag(s24 > 2000, s24)
    df["foreign"] = _flag(df["country"] != "SG", df["country"])
    # New-device interactions (new device abroad / at night / on a young account / for a big amount)
    df["nd_foreign"] = df["new_device"] * df["foreign"]
    df["nd_night"] = df["new_device"] * df["night"]
    df["nd_young"] = df["new_device"] * df["young_account"]
    df["nd_high_amount"] = df["new_device"] * df["high_amount"]
    # High-risk merchant categories, card-not-present and big-ticket combinations
    risky = df["merchant_category"].isin(["cash_transfer", "electronics", "luxury"])
    df["risky_cat"] = _flag(risky, df["merchant_category"])
    df["risky_cat_cnp"] = _flag(risky & (df["transaction_channel"] != "card_present"),
                                df["merchant_category"], df["transaction_channel"])
    df["risky_cat_high_amount"] = df["risky_cat"] * df["high_amount"]
    return df


def _flag(cond, *sources):
    """0/1 flag that is NaN wherever any source value is missing."""
    out = pd.Series(np.asarray(cond, dtype=float), index=sources[0].index)
    for s in sources:
        out[s.isna()] = np.nan
    return out
