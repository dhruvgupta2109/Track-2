import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
import numpy as np, lrlab
from evaluate import evaluate; from lrlab import F
# ablate inside the new-device family (quick views first)
orig = lrlab.add_inter
def only(keep):
    def f(d, fams):
        d, cols = orig(d, fams); return d, [c for c in cols if c in keep]
    return f
for keep in [["nd_x_logage"], ["nd_x_amt"], ["nd_x_foreign"], ["nd_x_night"], ["nd_x_logage", "nd_x_amt"], ["nd_x_logage", "nd_x_amt", "nd_x_night"]]:
    lrlab.add_inter = only(keep); evaluate("v3+" + "+".join(keep), F(fams=("nd",)), quick=True)
lrlab.add_inter = orig
