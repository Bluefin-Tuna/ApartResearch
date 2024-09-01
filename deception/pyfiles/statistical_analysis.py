import numpy as np
import pandas as pd
from scipy.special import kl_div
from scipy.stats import chi2_contingency, anderson_ksamp, ks_2samp

def compute_kl_divergence(control, experiment):
    data1 = pd.read_csv(control)
    data2 = pd.read_csv(experiment)
    
    outcomes1 = data1['outcome'].value_counts(normalize=True).sort_index()
    outcomes2 = data2['outcome'].value_counts(normalize=True).sort_index()
    
    all_outcomes = outcomes1.index.union(outcomes2.index)
    outcomes1 = outcomes1.reindex(all_outcomes, fill_value=0)
    outcomes2 = outcomes2.reindex(all_outcomes, fill_value=0)
    
    kl_divergence = np.sum(kl_div(outcomes1, outcomes2))
    
    return kl_divergence

def chi_squared_test(control, experiment):
    data1 = pd.read_csv(control)
    data2 = pd.read_csv(experiment)
    
    contingency_table = pd.crosstab(data1['outcome'], data2['outcome'])
    
    chi2, p_value, _, _ = chi2_contingency(contingency_table)
    
    return chi2, p_value

def anderson_darling_test(control, experiment):
    data1 = pd.read_csv(control)
    data2 = pd.read_csv(experiment)
    
    outcomes1 = data1['outcome'].values
    outcomes2 = data2['outcome'].values
    
    result = anderson_ksamp([outcomes1, outcomes2])
    
    return result.statistic, result.pvalue

def kolmogorov_smirnov_test(control, experiment):
    data1 = pd.read_csv(control)
    data2 = pd.read_csv(experiment)
    
    outcomes1 = data1['outcome'].values
    outcomes2 = data2['outcome'].values
    
    ks_statistic, ks_pvalue = ks_2samp(outcomes1, outcomes2)
    
    return ks_statistic, ks_pvalue

