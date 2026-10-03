import pandas as pd, numpy as np
pd.set_option("display.width",200); pd.set_option("display.max_columns",50)
tr=pd.read_csv("train.csv"); te=pd.read_csv("test.csv")
print(tr.shape, te.shape); print(tr.dtypes)
print("fraud rate", tr.fraud.mean(), tr.fraud.sum())
print("missing train\n", tr.isna().mean().round(4)); print("missing test\n", te.isna().mean().round(4))
print(tr.describe().T)
for c in ["merchant_category","country","transaction_channel","new_device","transaction_hour"]:
    g=tr.groupby(c,dropna=False).fraud.agg(["size","mean"]); g["test_n"]=te[c].value_counts(dropna=False).reindex(g.index)
    print(g)
print("nunique", tr.nunique())
f=[c for c in tr.columns if c not in("id","fraud")]
print("dup feature rows train", tr.duplicated(f).sum(), "test", te.duplicated(f).sum(), "train-test overlap", len(tr[f].merge(te[f])))
print("id numeric range", tr.id.str[2:].astype(int).agg(["min","max"]).tolist(), te.id.str[2:].astype(int).agg(["min","max"]).tolist())
tr["idn"]=tr.id.str[2:].astype(int)
print("fraud by id decile", tr.groupby(pd.qcut(tr.idn,10)).fraud.mean().round(4).tolist())
for c in ["transaction_amount","transactions_last_24h","spend_last_24h","account_age","transactions_last_1h"]:
    print(c, "fraud by decile", tr.groupby(pd.qcut(tr[c],10,duplicates="drop")).fraud.mean().round(4).tolist())
    print("  missing-rate fraud", tr[tr[c].isna()].fraud.mean() if tr[c].isna().any() else None)
print("isna pattern vs fraud"); print(tr.assign(nmiss=tr.isna().sum(1)).groupby("nmiss").fraud.agg(["size","mean"]))
