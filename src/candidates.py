"""Registry of every candidate with strict 5x5 OOF (evaluate.SPLITS, seed 2026) and a way to (re)generate test predictions."""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import models, r5lab, r6lab, lrlab
from build_v5 import fit_v5, predict_v5

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _sub(name):
    return lambda tr, te: pd.read_csv(os.path.join(ROOT, "submissions", f"sub_{name}.csv")).prediction.values


def _fp(fp):
    return lambda tr, te: fp(tr, te)


def _lazy(modname, attr, *args):
    def f(tr, te):
        mod = __import__(modname); g = getattr(mod, attr)
        return (g(*args) if args else g)(tr, te)
    return f


def _nd2(fams=("nd",)):
    orig = lrlab.add_inter
    def fp(a, b):
        def f(d, f_):
            d, cols = orig(d, f_)
            return d, [c for c in cols if not c.startswith("nd_") or c in ("nd_x_logage", "nd_x_night")]
        lrlab.add_inter = f
        try: return lrlab.fit_predict(a, b, fams=fams)
        finally: lrlab.add_inter = orig
    return fp


# name: (oof file stem, family, test-pred generator, deps, previous verdict / reason)
REG = {
    "v0_lr":               ("lr_raw_log", "LR raw", _sub("v0_lr"), "sklearn", "baseline"),
    "v1_lr_eng":           ("lr_eng_noind_C0.1", "LR eng", _sub("v1_lr_eng"), "sklearn", "superseded"),
    "v3_lr":               ("oof5_v3_lr", "LR eng", _sub("v3_lr"), "sklearn", "superseded"),
    "v4_lr":               ("oof5_v4", "LR + nd interactions", _sub("v4_lr"), "sklearn", "fallback"),
    "v5_lr_knn":           ("oof5_v5_prod", "LR + age/ctx/kNN", _sub("v5_lr_knn"), "sklearn", "current best"),
    "v5a_lr":              ("oof5_v5a_prod", "LR + age/ctx", _sub("v5a_lr"), "sklearn", "not submitted: assumed unresolvable vs v5"),
    "v5b_burstnight":      ("oof5_v5b_burstnight", "LR + burst x night", _sub("v5b_burstnight_probe"), "sklearn", "rejected: holdout collapse"),
    "v4_burst_nd":         ("oof5_v3+burst+nd", "LR + burst interactions", _fp(lambda a, b: lrlab.fit_predict(a, b, fams=("burst", "nd"))), "sklearn", "rejected: holdout collapse (best hnW of round 4)"),
    "v4_cross_nd_hour":    ("oof5_v4_cross_nd_hour", "LR + nd x hour", None, "sklearn", "within noise"),
    "v4_knn_full":         ("oof5_r6_knn", "LR + kNN (4 feats)", _fp(r6lab.F(gens=("knn",))), "sklearn", "holdout drop"),
    "v4_knn_dratio":       ("oof5_r6_knn_dratio_full", "LR + kNN ratio", _fp(r6lab.F(gens=("knn",), keep=["d_ratio"])), "sklearn", "fresh seeds small"),
    "v4_ctxamt":           ("oof5_r6_ctxamt", "LR + ctx amount", _fp(r6lab.F(gens=("ctxamt",))), "sklearn", "small"),
    "catboost_d4":         ("oof5_catboost_d4", "CatBoost", _lazy("exp_r4d", "cb"), "catboost", "weaker, holdout collapse"),
    "lgb_mono":            ("oof5_lgb_mono", "LightGBM monotone", _lazy("exp_r4d", "lg"), "lightgbm", "weaker std, best holdout_hnW"),
    "extratrees":          ("oof5_base_et", "ExtraTrees", _lazy("exp_r6d_bases", "et"), "sklearn", "weak standalone"),
    "randomforest":        ("oof5_base_rf", "RandomForest", _lazy("exp_r6d_bases", "rf"), "sklearn", "weak standalone"),
    "histgb":              ("oof5_base_hgb", "HistGB", _lazy("exp_r6d_bases", "hgb"), "sklearn", "weak standalone"),
    "subtype_young":       ("oof5_base_st_young", "subtype detector", _lazy("exp_r6d_bases", "subtype", "young"), "sklearn", "weak std, hnW ~v4"),
    "subtype_device":      ("oof5_base_st_device", "subtype detector", _lazy("exp_r6d_bases", "subtype", "device"), "sklearn", "weak"),
    "moe_newdevice":       ("oof5_r6_moe_newdevice", "MoE nd", _fp(lambda a, b: r6lab.moe(a, b, "new_device")), "sklearn", "below v4"),
}


def load_oof(stem):
    p = os.path.join(ROOT, "experiments", (stem if stem.startswith("oof") else "oof_" + stem) + ".npy")
    if not os.path.exists(p): p = os.path.join(ROOT, "experiments", "oof5_" + stem + ".npy")
    o = np.load(p)
    return o[:5] if o.ndim == 2 else o
