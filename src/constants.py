import numpy as np
import pandas as pd
from features_prod import NUM_COLS


def fit_constants(train):
    """Target-independent constants learned from the training rows only."""
    num = train[NUM_COLS].apply(pd.to_numeric, errors="coerce")
    return {"cat_med_amt": train.groupby("merchant_category").transaction_amount.median().to_dict(),
            "global_med_amt": float(train.transaction_amount.median()),
            "log_age_med": float(np.log1p(num["account_age"]).median()),
            "num_lo": num.min().to_dict(), "num_hi": num.max().to_dict()}
