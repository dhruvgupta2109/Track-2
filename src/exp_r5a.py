import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
from evaluate import evaluate; from r5lab import F
import evaluate as E
for add in [["burst_young"], ["burst_night"], ["burst_mature"], ["burst_x_logage"], ["day_burst_mature"], ["domestic_burst"], ["cash_burst"],
            ["big_mature"], ["big_young"], ["amt_pos_x_mature"], ["cash_mature"], ["young_night"], ["foreign_young"], ["foreign_nd"],
            ["t1_ge3"], ["t1_hi"], ["spike_avg"]]:
    evaluate("v4+" + "+".join(add), F(add), quick=True, ref="v4")
