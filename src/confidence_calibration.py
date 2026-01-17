import numpy as np
import pandas as pd
from scipy.stats import spearmanr

def compute_confidence(rank):
    """Confidence based on reciprocal rank"""
    return 1 / rank if rank > 0 else 0.0

def expected_calibration_error(confidences, correctness, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    confidences = np.array(confidences)
    correctness = np.array(correctness)

    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
        if mask.sum() == 0:
            continue

        bin_acc = correctness[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.sum() / len(confidences)) * abs(bin_acc - bin_conf)

    return round(ece, 4)

def confidence_correlation(confidences, correctness):
    corr, _ = spearmanr(confidences, correctness)
    return round(corr, 4)
