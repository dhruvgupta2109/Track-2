import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
import models
from evaluate import evaluate
evaluate("v4_prod", lambda a, b: models.predict_bundle(models.fit_lr(a, C=0.1, num=models.LR_NUM_V4), b))
