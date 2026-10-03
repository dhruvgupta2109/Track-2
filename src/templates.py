"""Exact boxes of the three planted test templates (uniform synthetic boxes, never in hours 5,6,7,21,22)."""
import numpy as np, pandas as pd

def _in(s, vals): return s.isin(vals).to_numpy()

def template_masks(df, loose_missing=True):
    """returns dict name -> bool mask. A missing value in a box column counts as inside the box (test has extra MCAR blanks)."""
    def ok(col, cond):
        c = df[col]; m = cond(c)
        m = np.asarray(m, bool)
        return m | c.isna().to_numpy() if loose_missing else m
    h = df.transaction_hour
    day = h.between(8, 20).to_numpy(); night = h.isin([23, 0, 1, 2, 3, 4]).to_numpy()
    t1 = ok("transactions_last_1h", lambda c: c <= 1)
    T1 = (day & ok("new_device", lambda c: c == 1) & ok("merchant_category", lambda c: c.isin(["grocery", "retail", "transport", "utilities"]))
          & ok("country", lambda c: c == "SG") & ok("transaction_channel", lambda c: c.isin(["card_present", "mobile_app"]))
          & ok("transactions_last_24h", lambda c: c.between(1, 5)) & t1 & (df.transaction_amount.between(10, 180).to_numpy())
          & ok("spend_last_24h", lambda c: c <= 450) & ok("account_age", lambda c: c >= 750))
    T2 = (day & ok("new_device", lambda c: c == 0) & ok("merchant_category", lambda c: c.isin(["cash_transfer", "electronics", "luxury", "travel"]))
          & ok("country", lambda c: c.isin(["AU", "JP", "MY", "SG"])) & ok("transaction_channel", lambda c: c.isin(["bank_transfer", "card_present", "mobile_app"]))
          & ok("transactions_last_24h", lambda c: c.between(1, 4)) & t1 & (df.transaction_amount.between(1500, 6000).to_numpy())
          & ok("spend_last_24h", lambda c: c.between(100, 2000)) & ok("account_age", lambda c: c >= 1100))
    T3 = (night & ok("new_device", lambda c: c == 0) & ok("merchant_category", lambda c: c.isin(["food_delivery", "gaming", "grocery", "transport"]))
          & ok("country", lambda c: c == "SG") & ok("transaction_channel", lambda c: c.isin(["card_present", "mobile_app"]))
          & ok("transactions_last_24h", lambda c: c.between(1, 6)) & t1 & (df.transaction_amount.between(5, 90).to_numpy())
          & ok("spend_last_24h", lambda c: c <= 500) & ok("account_age", lambda c: c >= 700))
    return {"T1": T1, "T2": T2, "T3": T3}

if __name__ == "__main__":
    tr = pd.read_csv("train.csv"); te = pd.read_csv("test.csv")
    Mte, Mtr = template_masks(te), template_masks(tr)
    for k in Mte:
        # base rows expected inside the box: train box rate * 8960 base rows (train has the same base generator up to drift)
        exp_base = Mtr[k].mean() * 8960
        print(f"{k}: test {Mte[k].sum()}, train {Mtr[k].sum()} (fraud {tr.fraud.values[Mtr[k]].sum()}, rate {tr.fraud.values[Mtr[k]].mean():.4f}); "
              f"expected base rows in test box ~{exp_base:.0f} -> template purity ~{1 - exp_base / Mte[k].sum():.2f}")
