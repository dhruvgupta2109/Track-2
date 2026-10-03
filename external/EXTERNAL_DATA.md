# External data ledger (Track 2)

Handbook: "You can however, curate your own datasets or use publicly available ones..." -> external data permitted.

## 1. gh_shajidh_transactions.csv  (RESEARCH ONLY - NOT USED IN ANY PACKAGE)
- Source: https://github.com/shajidhshaik90143-ux/Fraud-Transaction-Detection-System (data/transactions.csv, src/generate_data.py)
- License: none stated (repo created 2026-09-28). Downloaded 2026-10-03. Kept only for the transfer experiment.
- Content: 12,000 synthetic rows, fraud 12.0%; labels are drawn from a hand-written logistic risk function (see gh_shajidh_generate_data.py).
- Not a copy of the organisers' data: different categories (restaurant/fuel/jewelry...), different schema, different fraud rate.
- Mapping used: amount->transaction_amount, hour, merchant (cash_withdrawal->cash_transfer, jewelry->luxury, online_shopping/pharmacy->retail,
  restaurant->food_delivery, fuel->transport), is_international->country SG/US, card_present->channel card_present/ecommerce,
  transactions_last_24h, account_age_days, is_new_device; spend_last_24h ~ avg_amount_30d*tx24; transactions_last_1h missing.
- Domain classifier (competition train vs external): ROC-AUC 0.967 -> direct augmentation risky.
- Weighted augmentation (validated ONLY on competition folds, v4 features + source flag): w=0.05 -0.0007, 0.1 -0.0033, 0.25 -0.0100, 0.5 -0.0164 PR-AUC. Rejected.
- Used for: hypothesis generation only. Its generator style (additive logit of threshold indicators) motivated the
  threshold-basis search that found the account_age<120 step used in v5 (validated purely on competition data).

## Not pursued
- IEEE-CIS / PaySim / ULB credit-card: require Kaggle auth and share at most amount/hour semantics; labels come from different
  (real or simulated) processes and cannot inform the organisers' synthetic label function. Domain mismatch would be ~1.0.
