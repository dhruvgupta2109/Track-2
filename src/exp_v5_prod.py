import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
from evaluate import evaluate
from build_v5 import fit_v5, predict_v5
evaluate("v5_prod", lambda a, b: predict_v5(fit_v5(a, True), b), quick=True, ref="v4")
evaluate("v5a_prod", lambda a, b: predict_v5(fit_v5(a, False), b), quick=True, ref="v4")
