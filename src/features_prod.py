import numpy as np
import pandas as pd

# Canonical row-local feature code. This exact text is pasted into the prediction notebook,
# and training/validation import it too, so there is one source of truth (no train/inference skew).
CAT_COLS = ["merchant_category", "country", "transaction_channel"]
NUM_COLS = ["transaction_amount", "transaction_hour", "transactions_last_24h", "spend_last_24h",
            "account_age", "new_device", "transactions_last_1h"]


def sanitize(df, K):
    """Make any input schema-safe: missing columns -> NaN, non-numeric -> NaN, numerics clipped to the
    training range, category strings normalised. Never raises on odd values."""
    df = df.copy()
    for c in CAT_COLS + NUM_COLS:
        if c not in df.columns:
            df[c] = np.nan
    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).clip(K["num_lo"][c], K["num_hi"][c])
    df["transaction_hour"] = df["transaction_hour"].round()
    df["new_device"] = df["new_device"].round()
    for c in CAT_COLS:
        s = df[c].astype(object)
        s = s.map(lambda v: np.nan if v is None or (isinstance(v, float) and np.isnan(v)) else str(v).strip())
        df[c] = s.map(lambda v: np.nan if not isinstance(v, str) or v in ("", "nan", "None", "NaN") else (v.upper() if c == "country" else v.lower()))
    return df


def make_features(df, K):
    df = sanitize(df, K)
    amt, sp, t24, t1, age = (df.transaction_amount, df.spend_last_24h, df.transactions_last_24h,
                             df.transactions_last_1h, df.account_age)
    med = df.merchant_category.map(K["cat_med_amt"]).astype(float).fillna(K["global_med_amt"])
    avg = (sp / t24.where(t24 > 0)).replace([np.inf, -np.inf], np.nan)
    df["log_amt"] = np.log1p(amt)
    df["amt_rel_cat"] = np.log1p(amt) - np.log1p(med)       # amount vs typical amount for the merchant category
    df["log_spend"] = np.log1p(sp)
    df["avg_spend"] = np.log1p(avg)                           # account's average spend per tx in last 24h
    df["amt_rel_avg"] = np.log1p(amt) - np.log1p(avg)         # this amount vs that recent average
    df["log_age"] = np.log1p(age)
    df["young"] = (age < 90).astype(float).where(age.notna())
    df["t1_hi"] = np.clip(t1 - 2, 0, None)                    # velocity matters only above ~2 tx/hour
    df["t24_hi"] = np.clip(t24 - 9, 0, None)                  # ... and above ~9 tx/day
    df["burst"] = (t24 >= 16).astype(float).where(t24.notna())
    df["t1_share"] = (t1 / t24.where(t24 > 0)).clip(0, 1)
    # rapid-fire (>=4 tx in the last hour) outside the 24h burst cluster; inside a burst the 1h count adds nothing
    df["t1_ge4_nb"] = ((t1 >= 4) & (df["burst"] != 1)).astype(float).where(t1.notna())
    df["night"] = df.transaction_hour.isin([23, 0, 1, 2, 3]).astype(float)
    df["foreign"] = (df.country != "SG").astype(float).where(df.country.notna())
    # v4: new-device risk depends on account age and time of day (validated on all CV views); centred at log(665 days)
    nd = df.new_device.fillna(0)
    df["nd_x_logage"] = nd * (df.log_age.fillna(K.get("log_age_med", 6.67)) - 6.5)
    df["nd_x_night"] = nd * df.night
    df["hc"] = df.transaction_hour.map(lambda h: np.nan if pd.isna(h) else str(int(h)))
    return df


# ---------------- v5 additions (row-local given stored train constants / reference sets) ----------------
V5_EMB = ["log_amt", "amt_rel_cat", "log_age", "new_device", "transactions_last_24h", "transactions_last_1h", "night", "foreign", "avg_spend", "amt_rel_avg"]


def add_v5_features(df, K, knn=True):
    """age step, amount relative to channel/country medians, amount percentile within merchant category,
    and (optionally) the kNN distance ratio to stored training fraud / legit reference points."""
    df["age_lt120"] = (df.account_age < 120).astype(float).where(df.account_age.notna())
    for g in ["transaction_channel", "country"]:
        df["amt_rel_" + g] = df.log_amt - df[g].map(K["med_log_amt_" + g]).astype(float).fillna(K["global_med_log_amt"])
    pc = np.full(len(df), 0.5)
    for cat, v in K["sorted_log_amt_by_cat"].items():
        m = (df.merchant_category == cat).values
        if m.any():
            pc[m] = np.searchsorted(v, df.log_amt.values[m]) / len(v)
    df["amt_pct_cat"] = pc
    if knn:
        E = (df[V5_EMB].fillna(K["emb_median"]).values - K["emb_mean"]) / K["emb_scale"]
        d_f = K["nn_fraud"].kneighbors(E, n_neighbors=1)[0][:, 0]
        d_l = K["nn_legit"].kneighbors(E, n_neighbors=1)[0][:, 0]
        df["d_ratio"] = np.log1p(d_f) - np.log1p(d_l)
    return df
