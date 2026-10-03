def make_features(df):
    """Same row-local feature engineering as training."""
    df = df.copy()
    for c in ["transaction_amount", "spend_last_24h", "account_age"]:
        df[c] = np.log1p(df[c])
    return df
