import sys, numpy as np, pandas as pd
sys.path.insert(0, "src")
from inject import segs, Dtr, Dte, tr, te, CLEAN
NIGHT, DAY = [23, 0, 1, 2, 3, 4], list(range(8, 21))
Str, Ste = segs(Dtr), segs(Dte); htr, hte = Dtr.transaction_hour.values, Dte.transaction_hour.values
ctr, cte = np.isin(htr, CLEAN), np.isin(hte, CLEAN)
rows = []
for k in Str:
    if k == "night(23-4)": continue
    st, se = np.asarray(Str[k], bool), np.asarray(Ste[k], bool); c_s = (se & cte).sum(); out = [k, se.sum()]
    for band in (NIGHT, DAY):
        btr, bte = np.isin(htr, band), np.isin(hte, band)
        r = (st & btr).sum() / max((st & ctr).sum(), 1); inj = (se & bte).sum() - r * c_s
        out += [round(inj), round(inj / max((se & bte).sum(), 1), 2)]
    out.append(round(tr.fraud.values[st].mean(), 4)); rows.append(out)
df = pd.DataFrame(rows, columns=["segment", "n_test", "inj_NIGHT", "inj_frac_NIGHT", "inj_DAY", "inj_frac_DAY", "train_fraud_rate"])
pd.set_option("display.width", 200); print(df.to_string(index=False))
# hour profile of excess: per hour excess over base level
hc = pd.Series(hte).value_counts().sort_index(); base = hc[CLEAN].mean(); print("excess per hour:", (hc - base).round(0).astype(int).to_dict())
