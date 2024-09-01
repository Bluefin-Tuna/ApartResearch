import numpy as np
import pandas as pd
from scipy.special import kl_div
from scipy.stats import chisquare, anderson_ksamp, ks_2samp, PermutationMethod
from scipy.spatial import distance
from collections import Counter
import json

def parse_frequencies(data):
    if type(data[0]) == dict:
        freq = Counter()
        for i in range(len(data)):
            freq += data[i]
        freq = pd.Series(freq).sort_index()
        freq = freq / freq.sum()
        return freq
    else:
        return data.value_counts(normalize=True).sort_index()

def compute_kl_divergence(control, experiment, feature):
    outcomes1, outcomes2 = parse_frequencies(control[feature]), parse_frequencies(experiment[feature])
    return np.sum(kl_div(outcomes1, outcomes2))

def compute_jensenshannon_distance(control, experiment, feature):
    outcomes1, outcomes2 = parse_frequencies(control[feature]), parse_frequencies(experiment[feature])
    return distance.jensenshannon(outcomes1, outcomes2)

def chi_squared_test(control, experiment, feature):
    outcomes1, outcomes2 = parse_frequencies(control[feature]), parse_frequencies(experiment[feature])
    chi2, p_value = chisquare(outcomes2, outcomes1)
    return chi2, p_value

def anderson_darling_test(control, experiment, feature):
    outcomes1, outcomes2 = parse_frequencies(control[feature]), parse_frequencies(experiment[feature])

    result = anderson_ksamp([outcomes1, outcomes2], method=PermutationMethod())

    return result.statistic, result.significance_level

def kolmogorov_smirnov_test(control, experiment, feature):
    outcomes1, outcomes2 = parse_frequencies(control[feature]), parse_frequencies(experiment[feature])
    ks_statistic, ks_pvalue = ks_2samp(outcomes1, outcomes2)
    return ks_statistic, ks_pvalue