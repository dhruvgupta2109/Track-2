import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
import models
from features_prod import make_features; from constants import fit_constants
from common import load
pd.set_option("display.width", 250)
tr, te = load(); y = tr.fraud.values
K = fit_constants(tr); D = make_features(tr, K)
def oof_without(drop_num=(), drop_cat=()):
    num = [c for c in models.LR_NUM_V4 if c not in drop_num]; cat = [c for c in models.LR_CAT if c not in drop_cat]; o = np.zeros(len(y))
    for i, j in StratifiedKFold(5, shuffle=True, random_state=0).split(D, y):
        o[j] = models.lr_pipeline(0.1, num, cat).fit(D.iloc[i][cat + num], y[i]).predict_proba(D.iloc[j][cat + num])[:, 1]
    return o
def show(var, bins, drop_num=(), drop_cat=(), x=None):
    o = oof_without(drop_num, drop_cat); x = D[var] if x is None else x
    g = pd.DataFrame({"b": pd.cut(x, bins), "y": y, "p": o}).groupby("b", observed=True).agg(n=("y", "size"), obs=("y", "sum"), exp=("p", "sum"))
    g["obs/exp"] = (g.obs / g.exp).round(2); g["rate"] = (g.obs / g.n).round(4)
    print(f"\n### {var} (model without its terms)"); print(g.round(1).T.to_string())
show("account_age", [0, 15, 30, 45, 60, 75, 90, 120, 180, 270, 365, 545, 730, 1095, 1500, 2000, 3000, 4001], drop_num=("log_age", "young", "nd_x_logage"))
show("amt_rel_avg", [-9, -1, -0.5, 0, 0.25, 0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2, 3, 9], drop_num=("amt_rel_avg",))
D["amt_over_avg"] = D.transaction_amount / (D.spend_last_24h / D.transactions_last_24h)
show("amt_over_avg", [0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 8, 1e9], drop_num=("amt_rel_avg",), x=D.amt_over_avg)
show("transaction_amount", [0, 10, 25, 50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000, 1e5], drop_num=("log_amt", "amt_rel_cat", "amt_rel_avg"))
show("transactions_last_24h", [0, 2, 4, 6, 7, 8, 9, 10, 11, 12, 13, 15, 20, 30], drop_num=("transactions_last_24h", "t24_hi", "burst"))
show("transactions_last_1h", [-1, 0, 1, 2, 3, 4, 5, 6, 10], drop_num=("t1_ge4_nb",))
show("spend_last_24h", [-1, 50, 100, 200, 400, 700, 1000, 1500, 2000, 3000, 5000, 1e6], drop_num=("log_spend", "avg_spend", "amt_rel_avg"))
show("transaction_hour", list(np.arange(-1, 24)), drop_cat=("hc",), drop_num=("night", "nd_x_night"))
