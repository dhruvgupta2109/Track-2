import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from evaluate import evaluate, TR
import models
from features_prod import make_features; from constants import fit_constants
ex = pd.read_csv("external/gh_shajidh_transactions.csv")
mm = {"grocery": "grocery", "electronics": "electronics", "travel": "travel", "gaming": "gaming", "cash_withdrawal": "cash_transfer", "jewelry": "luxury",
      "online_shopping": "retail", "restaurant": "food_delivery", "fuel": "transport", "pharmacy": "retail"}
E = pd.DataFrame({"transaction_amount": ex.amount, "transaction_hour": ex.transaction_hour, "merchant_category": ex.merchant_category.map(mm),
                  "country": np.where(ex.is_international == 1, "US", "SG"), "transaction_channel": np.where(ex.card_present == 1, "card_present", "ecommerce"),
                  "transactions_last_24h": ex.transactions_last_24h.clip(lower=1), "spend_last_24h": ex.avg_amount_30d * ex.transactions_last_24h.clip(lower=1),
                  "account_age": ex.account_age_days.clip(upper=4000), "new_device": ex.is_new_device, "transactions_last_1h": np.nan, "fraud": ex.is_fraud})
print("external rows", len(E), "fraud rate", E.fraud.mean().round(4))
# domain classifier competition-train vs external (mapped columns)
cols = ["transaction_amount", "transaction_hour", "transactions_last_24h", "account_age", "new_device", "merchant_category", "country", "transaction_channel"]
A = pd.concat([TR[cols], E[cols]]); z = np.r_[np.zeros(len(TR)), np.ones(len(E))]
for c in ["merchant_category", "country", "transaction_channel"]: A[c] = A[c].astype("category")
p = cross_val_predict(lgb.LGBMClassifier(n_estimators=200, verbose=-1), A, z, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
print("domain AUC competition vs external:", round(roc_auc_score(z, p), 4))
def aug(w, with_flag=True):
    def fp(a, b):
        K = fit_constants(a); Aa = make_features(a, K); Ee = make_features(E, K); Bb = make_features(b, K)
        num = list(models.LR_NUM_V4) + (["src_ext"] if with_flag else [])
        Aa["src_ext"] = 0.0; Ee["src_ext"] = 1.0; Bb["src_ext"] = 0.0
        X = pd.concat([Aa, Ee]); y = np.r_[a.fraud.values, E.fraud.values]; sw = np.r_[np.ones(len(Aa)), np.full(len(Ee), w)]
        m = models.lr_pipeline(0.1, num, models.LR_CAT)
        m.fit(X[models.LR_CAT + num], y, logisticregression__sample_weight=sw)
        return m.predict_proba(Bb[models.LR_CAT + num])[:, 1]
    return fp
for w in [0.05, 0.1, 0.25, 0.5]:
    evaluate(f"r7_ext_aug_w{w}", aug(w), quick=True, ref="v4")
