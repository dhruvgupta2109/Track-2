import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
import lrlab
from evaluate import evaluate; from lrlab import F, BASE_NUM
orig = lrlab.add_inter
def nd2(d, fams):
    d, cols = orig(d, fams); return d, [c for c in cols if c in ("nd_x_logage", "nd_x_night")] + [c for c in cols if not c.startswith("nd_")]
lrlab.add_inter = nd2
evaluate("v4", F(fams=("nd",)))
for C in [0.03, 0.3, 1.0]: evaluate(f"v4_C{C}", F(fams=("nd",), C=C), quick=True)
evaluate("v4_l1_C0.3", F(fams=("nd",), C=0.3, penalty="l1"), quick=True)
evaluate("v4_en_C0.3", F(fams=("nd",), C=0.3, penalty="elasticnet", l1_ratio=0.5), quick=True)
evaluate("v4_spline_age_amt_k4", F(fams=("nd",), spline=(["log_age", "log_amt"], 4)), quick=True)
evaluate("v4_bins_age_amt_t24_q8", F(fams=("nd",), bins=(["log_age", "log_amt", "transactions_last_24h"], 8)), quick=True)
evaluate("v4_freq", F(fams=("nd",), freq=True), quick=True)
evaluate("v4_cross_merch_chan", F(fams=("nd",), cross=[("merchant_category", "transaction_channel")]), quick=True)
evaluate("v4_cross_nd_hour", F(fams=("nd",), cross=[("new_device", "hc")]), quick=True)
evaluate("v4_cross_chan_country", F(fams=("nd",), cross=[("transaction_channel", "country")]), quick=True)
