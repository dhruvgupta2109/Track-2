import sys; sys.path.insert(0, "src")
import lrlab
from exp_r4d import cb, lg
_orig = lrlab.add_inter
def _nd2(d, fams):
    d, cols = _orig(d, fams); return d, [c for c in cols if c in ("nd_x_logage", "nd_x_night")]
def v4(a, b):
    lrlab.add_inter = _nd2
    try: return lrlab.fit_predict(a, b, fams=("nd",))
    finally: lrlab.add_inter = _orig
