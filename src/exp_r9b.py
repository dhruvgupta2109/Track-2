import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
import numpy as np, pandas as pd
from r9lab import *
def with_cols(cols):
    return make(lambda A, B, y: v5_local(A, B, y, cols=cols))
for c in EMB:
    screen(f"drop {c}", with_cols([x for x in EMB if x != c]))
def prep_extra(D):
    D["hour_cos"] = np.cos(2 * np.pi * D.transaction_hour.fillna(12) / 24); D["hour_sin"] = np.sin(2 * np.pi * D.transaction_hour.fillna(12) / 24)
    D["bank"] = (D.transaction_channel == "bank_transfer").astype(float); D["ecom"] = (D.transaction_channel == "ecommerce").astype(float)
    D["cash"] = (D.merchant_category == "cash_transfer").astype(float); D["n_miss"] = D[["merchant_category", "country", "transaction_channel", "transactions_last_24h", "spend_last_24h", "account_age", "new_device", "transactions_last_1h"]].isna().sum(1)
    return D
def add_cols(extra):
    def fn(A, B, y):
        prep_extra(A); prep_extra(B); return v5_local(A, B, y, cols=EMB + extra)
    return make(fn)
for extra in [["hour_cos", "hour_sin"], ["log_spend"], ["young"], ["burst"], ["t1_ge4_nb"], ["bank", "ecom"], ["cash"], ["n_miss"], ["nd_x_logage"], ["age_lt120"], ["amt_pct_cat"], ["amt_rel_transaction_channel"]]:
    screen("add " + "+".join(extra), add_cols(extra))
