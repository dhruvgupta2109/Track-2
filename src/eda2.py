import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
pd.set_option("display.width",200); pd.set_option("display.max_columns",50)
tr=pd.read_csv("train.csv"); te=pd.read_csv("test.csv")
d=tr
print("1h>24h violations", (d.transactions_last_1h>d.transactions_last_24h).sum())
print("spend < amount", (d.spend_last_24h<d.transaction_amount).mean(), "fraud|spend<amt", d[d.spend_last_24h<d.transaction_amount].fraud.mean())
print("spend==0", (d.spend_last_24h==0).sum(), d[d.spend_last_24h==0].fraud.mean())
d["avg"]=d.spend_last_24h/d.transactions_last_24h; d["ratio"]=d.transaction_amount/(d.avg+1)
print("ratio deciles fraud", d.groupby(pd.qcut(d.ratio,10)).fraud.mean().round(3).tolist())
print("amount quantiles by fraud\n", d.groupby("fraud").transaction_amount.describe())
print(d.groupby("fraud")[["transactions_last_24h","transactions_last_1h","spend_last_24h","account_age","new_device"]].describe().T)
print("amount by category median\n", d.groupby("merchant_category").transaction_amount.median())
print("amount round?", (d.transaction_amount%1==0).mean(), d[d.transaction_amount%1==0].fraud.mean())
print("amount max values", d.transaction_amount.value_counts().head(8))
print("acct_age max count", (d.account_age==4000).sum(), (d.account_age==5).sum())
f=d[d.fraud==1]
print("fraud cats\n", f.merchant_category.value_counts(), f.country.value_counts(), f.transaction_channel.value_counts(), f.new_device.value_counts())
# how many fraud rows have no 'signal'
sig=(f.new_device==1)|(f.transactions_last_1h>=3)|(f.account_age<60)|(f.transaction_amount>800)|(f.transactions_last_24h>=10)
print("fraud with any obvious signal", sig.mean())
# adversarial validation
cols=[c for c in te.columns if c!="id"]
a=pd.concat([tr[cols],te[cols]]); y=np.r_[np.zeros(len(tr)),np.ones(len(te))]
for c in a.select_dtypes("object"): a[c]=a[c].astype("category")
p=cross_val_predict(lgb.LGBMClassifier(n_estimators=200,verbose=-1),a,y,cv=StratifiedKFold(5,shuffle=True,random_state=0),method="predict_proba")[:,1]
print("adversarial AUC", roc_auc_score(y,p))
m=lgb.LGBMClassifier(n_estimators=200,verbose=-1).fit(a,y); print(sorted(zip(m.booster_.feature_importance("gain").round(),cols),reverse=True))
a2=a.copy(); 
for c in cols: a2[c+"_na"]=a2[c].isna()
# without missingness
a3=a.fillna(a.median(numeric_only=True))
for c in a3.select_dtypes("category"): a3[c]=a3[c].cat.add_categories(["NA"]).fillna("NA") if a3[c].isna().any() else a3[c]
for c in ["transaction_amount","transactions_last_24h","spend_last_24h","account_age","transactions_last_1h"]:
    print(c, "train/test median", tr[c].median(), te[c].median(), " q90", tr[c].quantile(.9), te[c].quantile(.9), "q99", tr[c].quantile(.99), te[c].quantile(.99))
