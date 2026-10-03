import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
from evaluate import evaluate; from lrlab import F
evaluate("v3_lr", F())
for fam in [("burst",), ("nd",), ("big",), ("foreign",), ("sigcount",), ("burst", "nd"), ("burst", "nd", "big"), ("burst", "nd", "big", "foreign")]:
    evaluate("v3+" + "+".join(fam), F(fams=fam))
