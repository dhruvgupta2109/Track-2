"""Every public-scored file we still have on disk, with its public score and (where recorded) the portal hash (sha256[:10])."""
import os, hashlib, numpy as np, pandas as pd
R = ".."; SUR = f"{R}/Datathon Track 2 - suraj try"
HIST = [  # (path, public, recorded sha or None)
    ("submissions/sub_v3_lr.csv", 0.17206, "43744b16d0"), ("submissions/sub_v4_lr.csv", 0.17966, "aecf532c43"),
    ("submissions/sub_v4_lambda05_probe.csv", 0.16822, None), ("submissions/sub_v5_lr_knn.csv", 0.18626, "e9d43c2721"),
    ("submissions/sub_v5a_lr.csv", 0.18026, None), ("submissions/audit/cand_v4_knn_full.csv", 0.18436, None),
    ("submissions/audit/c1_v5_knn_catmetric.csv", 0.18388, None), ("submissions/audit/d1_v5_plus_1h_velocity.csv", 0.18463, None),
    ("submissions/audit/cand_lgb_mono.csv", 0.16962, None), ("submissions/audit/v6_E1_knn_metric.csv", 0.18237, "e60dcf1ca9"),
    ("submissions/audit/v6_rank90_skmlp10.csv", 0.18453, "dc84bc23cd"), ("submissions/final_try/E1_0.45k01_0.4surknn_0.15mlp.csv", 0.18053, "183aa749e8"),
    ("submissions/final_try/E2_0.4k01_0.15v5_0.3sur_0.15mlp.csv", 0.18398, "80d0c345d3"), ("submissions/final_try/E3_0.7k01_0.3sur.csv", 0.18662, "a3fbcb9c1b"),
    (f"{SUR}/submission.csv", 0.17617, "04aeb8a713"), (f"{SUR}/models/v2_lr_drift/submission.csv", 0.17136, None),
    (f"{SUR}/extra_submissions/cand1_v1_plus_paysim_score.csv", 0.17830, None), (f"{SUR}/extra_submissions/cand2_v1_iterative_imputation.csv", 0.17809, None),
    (f"{SUR}/extra_submissions/cand3_blend_0.8v1_0.2lgbm.csv", 0.17835, None),
    (f"{R}/FINAL_UPLOAD_v5_burst/predictions_v5_burst.csv", 0.198, "7f7182dc26"),
    ("submissions/shift/S1_burst-1.5.csv", 0.19293, "6e83e0fde2"), ("submissions/shift/S3_burst-1.0_ndold-0.5.csv", 0.20453, "f570586529"),
    ("submissions/shift/R1_burst-1.0_ndold-0.7_bigold-1.1.csv", 0.19707, "5887e82930"), ("submissions/shift/R2_burst-1.0_ndold-0.7_bigold-0.5.csv", 0.20171, "a448b752a7"),
    ("submissions/shift/G1_generator_formula.csv", 0.17795, "a2687f3a53"), ("submissions/shift/V1_S3_plus_travel-0.5.csv", 0.20237, "df00e35171"),
    ("submissions/shift/W1_E3base_S3shifts.csv", 0.19923, "6aa5f1f17b"), ("submissions/shift/W2_burst-1.0_ndold-0.75.csv", 0.20530, "31f108aa8c"),
    ("submissions/shift/S4_burst-1.0_ndold-1.0.csv", 0.20516, "7568ae0d7a"), ("submissions/shift/X1_burstOld-1.25_burstYoung0_ndold-0.75.csv", 0.18152, "208c516581"),
    ("submissions/shift/P1_burstYoung-2.0_burstOld-1.0_ndold-0.75.csv", 0.21289, "ce1e96a4f4"), ("submissions/shift/Q1_P1_plus_fastYoungND-1.0.csv", 0.20826, "b01f60b353"),
    ("submissions/shift/Q2_P1_burstYoung-3.0.csv", 0.20928, "91a8ac0a7d"), ("submissions/shift/U1_P1_bigold+0.5.csv", 0.21424, "c9c788e7f1"),
    ("submissions/shift/U2_P1_bigold+0.5_travel+0.5.csv", 0.21082, "e843505789"),
    (f"{R}/BEST_Z2_v5_youngburst-2_oldburst-0.5_ndold-0.75_0.21922/BEST_Z2_v5_youngburst-2_oldburst-0.5_ndold-0.75_0.21922/predictions_Z2_burstYoung-2.0_burstOld-0.5_ndold-0.75.csv", 0.21922, "95dccc5fd8"),
    ("submissions/shift/Z3_Z2_bigold+0.5.csv", 0.22215, "6234082ef2"), ("submissions/shift/Z4_Z2_bigold+0.5_oldburst-0.25.csv", 0.22195, "a6c7f33a09"),
    (f"{R}/Track2_K3_final_package_0.223/predictions_K3.csv", 0.223, "d03cb4d61b"), ("submissions/shift/K4_K3_fastYoungND+1.0.csv", 0.21477, "ee12bc40b6"),
    ("submissions/shift/K5_K3_fastYoungND+0.5.csv", 0.21741, "69e50b93c8"), ("submissions/shift/K6_K3_cleanhour_unshift.csv", 0.22041, "356731ccb5"),
    (f"{R}/Track2_M123_final_package_0.22523/predictions_M123.csv", 0.22523, None), (f"{R}/Track2_Q30_final_package_0.23288/predictions_Q30.csv", 0.23288, None),
    ("submissions/tmpl/C1_Q30_T2box_unshift.csv", 0.23277, "9422baa279"), ("submissions/tmpl/C7_Q30_T2box-1_hvout+0.5.csv", 0.22841, "bfc1046551"),
    ("submissions/tmpl/R2op_package_output.csv", 0.23434, "bc365689c1"), ("submissions/tmpl/R3op_package_output.csv", 0.23260, "7cdfaac8ee"),
]


def load(check=True):
    te_ids = pd.read_csv("test.csv").id.values; names, preds, obs = [], [], []
    for path, score, sha in HIST:
        if not os.path.exists(path): print("MISSING", path); continue
        h = hashlib.sha256(open(path, "rb").read()).hexdigest()[:10]
        if check and sha and h != sha: print(f"HASH MISMATCH {path}: file {h} recorded {sha}")
        d = pd.read_csv(path); assert (d.id.values == te_ids).all(), path
        names.append(os.path.basename(path)[:40]); preds.append(d.prediction.values.astype(float)); obs.append(score)
    return names, np.vstack(preds), np.array(obs)


if __name__ == "__main__":
    n, P, o = load(); print(len(n), "scored files loaded; public range", o.min(), o.max())
