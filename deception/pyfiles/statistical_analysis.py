import numpy as np
import pandas as pd
from scipy.special import kl_div
from scipy.stats import chisquare, chi2, anderson_ksamp, ks_2samp
from scipy.spatial import distance
from collections import Counter

SAMPLE_SIZE = 1000

def parse_frequencies(data, normalize=True):
    if isinstance(data[0], dict):
        freq = Counter()
        for d in data:
            freq += d
        freq = pd.Series(freq).sort_index()
        if normalize:
            freq = freq / freq.sum()
        return freq
    else:
        freq = data.value_counts(normalize=normalize).sort_index()
    return freq

def align_frequencies(freq1, freq2, normalize=True):
    all_indices = sorted(set(freq1.index).union(freq2.index))
    
    fill_value = 1e-8 if normalize else 5
    
    aligned_freq1 = freq1.reindex(all_indices, fill_value=fill_value)
    aligned_freq2 = freq2.reindex(all_indices, fill_value=fill_value)
    
    if normalize:
        aligned_freq1 = aligned_freq1.apply(lambda x: max(x, fill_value))
        aligned_freq2 = aligned_freq2.apply(lambda x: max(x, fill_value))
    else:
        aligned_freq1 = aligned_freq1.apply(lambda x: max(x, 5))
        aligned_freq2 = aligned_freq2.apply(lambda x: max(x, 5))
    
    total1 = aligned_freq1.sum()
    total2 = aligned_freq2.sum()
    
    if total1 != total2:
        scaling_factor = total1 / total2
        aligned_freq2 = aligned_freq2 * scaling_factor
    
    return aligned_freq1, aligned_freq2

def compute_kl_divergence(control, experiment, feature):
    control_data = control[feature].head(SAMPLE_SIZE)
    experiment_data = experiment[feature].head(SAMPLE_SIZE)
    
    outcomes1, outcomes2 = parse_frequencies(control_data, normalize=True), \
                           parse_frequencies(experiment_data, normalize=True)
    outcomes1, outcomes2 = align_frequencies(outcomes1, outcomes2, normalize=True)
    kl_div_result = np.sum(kl_div(outcomes1, outcomes2))
    return kl_div_result

def compute_jensenshannon_distance(control, experiment, feature):
    control_data = control[feature].head(SAMPLE_SIZE)
    experiment_data = experiment[feature].head(SAMPLE_SIZE)
    
    outcomes1, outcomes2 = parse_frequencies(control_data), parse_frequencies(experiment_data)
    outcomes1, outcomes2 = align_frequencies(outcomes1, outcomes2)
    
    js_distance = distance.jensenshannon(outcomes1, outcomes2)
    return js_distance

def chi_squared_test(control, experiment, feature, alpha=0.05):
    control_data = control[feature].head(SAMPLE_SIZE)
    experiment_data = experiment[feature].head(SAMPLE_SIZE)
    
    outcomes1, outcomes2 = parse_frequencies(control_data, normalize=False), \
                           parse_frequencies(experiment_data, normalize=False)
    outcomes1, outcomes2 = align_frequencies(outcomes1, outcomes2, normalize=False)
    
    degrees_of_freedom = len(outcomes1) - 1
    chi2_stat, p_value = chisquare(outcomes2, outcomes1)
    critical_value = chi2.ppf(1 - alpha, degrees_of_freedom)
    
    reject_null = chi2_stat > critical_value
    return chi2_stat, p_value, critical_value, reject_null

def anderson_darling_test(control, experiment, feature, alpha=0.05):
    control_data = control[feature].head(SAMPLE_SIZE)
    experiment_data = experiment[feature].head(SAMPLE_SIZE)
    outcomes1, outcomes2 = parse_frequencies(control_data), parse_frequencies(experiment_data)
    outcomes1, outcomes2 = align_frequencies(outcomes1, outcomes2)
    
    result = anderson_ksamp([outcomes1, outcomes2])
    
    critical_values = result.critical_values
    alpha_levels = [0.25, 0.10, 0.05, 0.025, 0.01, 0.005, 0.001]
    alpha_index = next((i for i, val in enumerate(alpha_levels) if alpha <= val), len(critical_values) - 1)
    critical_value = critical_values[alpha_index]
    
    reject_null = result.statistic > critical_value
  
    return result.statistic, result.pvalue, critical_value, reject_null

def kolmogorov_smirnov_test(control, experiment, feature):
    control_data = control[feature].head(SAMPLE_SIZE)
    experiment_data = experiment[feature].head(SAMPLE_SIZE)
    
    outcomes1, outcomes2 = parse_frequencies(control_data), parse_frequencies(experiment_data)
    outcomes1, outcomes2 = align_frequencies(outcomes1, outcomes2)

    ks_statistic, ks_pvalue = ks_2samp(outcomes1, outcomes2)
    return ks_statistic, ks_pvalue