def make_features(df, K):
    """Row-local feature engineering, identical to training (src/features.py). K = constants fitted on train."""
    df = df.copy()
    amt, sp, t24, t1, age = df.transaction_amount, df.spend_last_24h, df.transactions_last_24h, df.transactions_last_1h, df.account_age
    med = df.merchant_category.map(K["cat_med_amt"]).astype(float).fillna(K["global_med_amt"])
    df["log_amt"] = np.log1p(amt)
    df["amt_rel_cat"] = np.log1p(amt) - np.log1p(med)        # amount relative to typical amount for the merchant category
    df["log_spend"] = np.log1p(sp)
    df["avg_spend"] = np.log1p(sp / t24)                       # average spend per transaction in the last 24h
    df["amt_rel_avg"] = np.log1p(amt) - np.log1p(sp / t24)    # this amount vs the account's recent average
    df["log_age"] = np.log1p(age)
    df["young"] = (age < 90).astype(float).where(age.notna())
    df["t1_hi"] = np.clip(t1 - 2, 0, None)                     # velocity only matters above ~2 tx/hour
    df["t24_hi"] = np.clip(t24 - 9, 0, None)                   # ... and above ~9 tx/day
    df["burst"] = (t24 >= 16).astype(float).where(t24.notna())
    df["t1_share"] = t1 / t24
    df["night"] = df.transaction_hour.isin([23, 0, 1, 2, 3]).astype(float)
    df["foreign"] = (df.country != "SG").astype(float).where(df.country.notna())
    df["hc"] = df.transaction_hour.astype(str)
    return df
