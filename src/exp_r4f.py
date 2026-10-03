import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
import numpy as np, lrlab
from evaluate import evaluate
orig = lrlab.add_inter
def mk(extra, drop=()):
    def f(d, fams):
        d, cols = orig(d, ("nd", "big"))
        nd = d.new_device.fillna(0); la = d.log_age.fillna(d.log_age.median()); ar = d.amt_rel_cat.fillna(0)
        d["amt_cap1"] = d.amt_rel_cat.clip(upper=1.0)
        d["amt_pos_x_nd"] = nd * ar.clip(lower=0); d["amt_pos_x_logage"] = ar.clip(lower=0) * (la - 6.5)
        d["amt_pos_x_young"] = ar.clip(lower=0) * d.young.fillna(0)
        keep = ["nd_x_logage", "nd_x_night"] + extra
        return d, keep
    return f
def run(name, extra, drop=()):
    lrlab.add_inter = mk(extra)
    num = [c for c in lrlab.BASE_NUM if c not in drop]
    evaluate(name, lrlab.F(fams=("x",), num=num), quick=("quick" in name))
    lrlab.add_inter = orig
run("v4+amtpos_x_logage quick", ["amt_pos_x_logage"])
run("v4+amtpos_x_nd quick", ["amt_pos_x_nd"])
run("v4+amtpos_x_nd+logage quick", ["amt_pos_x_nd", "amt_pos_x_logage"])
run("v4+amtpos_x_young quick", ["amt_pos_x_young"])
run("v4_amtcap1+amtpos_x_nd+logage quick", ["amt_cap1", "amt_pos_x_nd", "amt_pos_x_logage"], drop=("amt_rel_cat",))
run("v4_amtcap1 quick", ["amt_cap1"], drop=("amt_rel_cat",))
