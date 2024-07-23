import pandas as pd
from scipy import stats

def perform_ks_test(control_data, experiment_data):
    """
    Perform Kolmogorov-Smirnov test between control and experiment data.

    Args:
        control_data (pd.Series): Control group data.
        experiment_data (pd.Series): Experiment group data.

    Returns:
        dict: A dictionary containing the D-statistic and p-value.
    """
    ks_statistic, p_value = stats.ks_2samp(control_data, experiment_data)
    return {
        'D-statistic': ks_statistic,
        'p-value': p_value
    }

def analyze_experiments(control_files, experiment_files):
    """
    Analyze experiments by performing K-S tests.

    Args:
        control_files (dict): Paths to control CSV files.
        experiment_files (dict): Paths to experiment CSV files.

    Returns:
        pd.DataFrame: Results of the K-S tests.
    """
    control_results = pd.read_csv(control_files['results'])
    control_dealer_draws = pd.read_csv(control_files['dealer_draws'])

    results = []

    for experiment_name, files in experiment_files.items():
        experiment_results = pd.read_csv(files['results'])
        experiment_dealer_draws = pd.read_csv(files['dealer_draws'])

        wins_ks_result = perform_ks_test(control_results['player_win'], experiment_results['player_win'])
        dealer_draws_ks_result = perform_ks_test(control_dealer_draws['card_value'], experiment_dealer_draws['card_value'])

        results.append({
            'Experiment': experiment_name,
            'D-statistic (wins)': wins_ks_result['D-statistic'],
            'p-value (wins)': wins_ks_result['p-value'],
            'D-statistic (dealer draws)': dealer_draws_ks_result['D-statistic'],
            'p-value (dealer draws)': dealer_draws_ks_result['p-value']
        })

    return pd.DataFrame(results)