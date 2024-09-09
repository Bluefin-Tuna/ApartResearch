from collections import Counter
import csv
import threading
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import seaborn as sns
import tqdm
from scipy import stats
from deception.environments.blackjack import Blackjack
from pyfiles.agent import *
from pyfiles.prompt import *
from pyfiles.statistical_analysis import *
from pyfiles.utils import random_draw_card
import ast

DATA_FOLDER = "results"

def run_experiment(num_games, draw_card_fn, unique_str):
    results = []
    pbar = tqdm.tqdm(range(num_games))

    for game_id in pbar:
        game = Blackjack(draw_card_fn)
        result = game.play()
        results.append(result)
        pbar.set_description(desc=f"Game {game_id}")

    results_df = pd.DataFrame(results)
    results_df["dealer_hand"] = results_df["dealer_hand"].apply(lambda x: dict(x))
    results_df["player_hand"] = results_df["player_hand"].apply(lambda x: dict(x))
    results_df.to_csv(f'{DATA_FOLDER}/{unique_str}_game_results.csv', index=False)

    summary_stats = {
        'total_games': num_games,
        'player_win_rate': results_df['player_win'].mean(),
        'dealer_bust_rate': results_df['dealer_bust'].mean(),
        'push_rate': results_df['push'].mean(),
        'avg_player_hand': results_df['player_hand_value'].mean(),
        'avg_dealer_hand': results_df['dealer_hand_value'].mean()
    }

    with open(f'{DATA_FOLDER}/{unique_str}_summary_stats.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for key, value in summary_stats.items():
            writer.writerow([key, value])
    
    sns.set_style("whitegrid")
    sns.set_palette("muted")

    plt.figure(figsize=(12, 6))
    sns.histplot(data=results_df, x='player_hand_value', fill=True, label='Player', binwidth=0.5, multiple='dodge')
    sns.histplot(data=results_df, x='dealer_hand_value', fill=True, label='Dealer', binwidth=0.5, multiple='dodge')
    plt.title('Distribution of Hand Values', fontsize=16)
    plt.xlabel('Hand Value', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.xticks(range(int(results_df['player_hand_value'].min()), int(results_df['player_hand_value'].max()) + 1))
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(f'{DATA_FOLDER}/{unique_str}_hand_value_distributions.png', dpi=300)
    plt.close()

    CARDS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'jack', 'queen', 'king', 'ace']
    dealer_hands = results_df['dealer_hand']
    player_hands = results_df['player_hand']
    dealer_card_freq = Counter()
    player_card_freq = Counter()
    for i in range(len(dealer_hands)):
        dealer_card_freq += dealer_hands[i]
        player_card_freq += player_hands[i]
    plt.figure(figsize=(12, 6))
    sns.barplot(x=CARDS, y=[dealer_card_freq[v] for v in CARDS], label='Dealer')
    sns.barplot(x=CARDS, y=[player_card_freq[v] for v in CARDS], label="Player")
    plt.title('Card Draw Frequency', fontsize=16)
    plt.xlabel('Card', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{DATA_FOLDER}/{unique_str}_card_frequency.png', dpi=300)
    plt.close()

    print("Experiment completed. Data saved to CSV files. Plots generated.")

def run_control_experiment(num_games, unique_str):
    run_experiment(num_games, random_draw_card, unique_str)

def run_agent_experiment(num_games, unique_str, agent, prompt):
    draw_card_fn = get_draw_card_fn(agent, prompt)
    run_experiment(num_games, draw_card_fn, unique_str)

def run_statistical_analysis(control_file, experiment_file):
    control_results_df = pd.read_csv(f'{DATA_FOLDER}/{control_file}')
    experiment_results_df = pd.read_csv(f'{DATA_FOLDER}/{experiment_file}')

    # run stat tests on card frequencies, hand values
    control_results_df['dealer_hand'] = control_results_df["dealer_hand"].apply(lambda x: ast.literal_eval(x))
    control_results_df["player_hand"] = control_results_df["player_hand"].apply(lambda x: ast.literal_eval(x))
    experiment_results_df["dealer_hand"] = experiment_results_df["dealer_hand"].apply(lambda x: ast.literal_eval(x))
    experiment_results_df["player_hand"] = experiment_results_df["player_hand"].apply(lambda x: ast.literal_eval(x))
  
    kl_div_dealer_card_freqs = compute_kl_divergence(control_results_df, experiment_results_df, 'dealer_hand')
    kl_div_player_card_freqs = compute_kl_divergence(control_results_df, experiment_results_df, 'player_hand')
    print("KL Divergence for Dealer Card Frequencies: ", kl_div_dealer_card_freqs)
    print("KL Divergence for Player Card Frequencies: ", kl_div_player_card_freqs)
    
    js_distance_dealer_card_freqs = compute_jensenshannon_distance(control_results_df, experiment_results_df, 'dealer_hand')
    js_distance_player_card_freqs = compute_jensenshannon_distance(control_results_df, experiment_results_df, 'player_hand')
    print("Jensen-Shannon Distance for Dealer Card Frequencies: ", js_distance_dealer_card_freqs)
    print("Jensen-Shannon Distance for Player Card Frequencies: ", js_distance_player_card_freqs)

    chi2_dealer_card_freqs, chi2_pval_dealer_card_freqs = chi_squared_test(control_results_df, experiment_results_df, 'dealer_hand')
    chi2_player_card_freqs, chi2_pval_player_card_freqs = chi_squared_test(control_results_df, experiment_results_df, 'player_hand')
    print('Chi-Squared Test for Dealer Card Frequences: ', chi2_dealer_card_freqs, chi2_pval_dealer_card_freqs)
    print('Chi-Squared Test for Player Card Frequences: ', chi2_player_card_freqs, chi2_pval_player_card_freqs)

    ks_dealer_card_freqs = kolmogorov_smirnov_test(control_results_df, experiment_results_df, 'dealer_hand')
    ks_player_card_freqs = kolmogorov_smirnov_test(control_results_df, experiment_results_df, 'player_hand')
    print("Kolmogorov-Smirnov Test for Dealer Card Frequencies: ", ks_dealer_card_freqs)
    print("Kolmogorov-Smirnov Test  for Player Card Frequencies: ", ks_player_card_freqs)
    
    print('================================')

    kl_div_dealer_hand_value = compute_kl_divergence(control_results_df, experiment_results_df, 'dealer_hand_value')
    kl_div_player_hand_value = compute_kl_divergence(control_results_df, experiment_results_df, 'player_hand_value')
    print("KL Divergence for Dealer Final Hand Values: ", kl_div_dealer_hand_value)
    print("KL Divergence for Player Final Hand Values: ", kl_div_player_hand_value)

    chi2_dealer_hand_value, chi2_pval_dealer_hand_value = chi_squared_test(control_results_df, experiment_results_df, 'dealer_hand_value')
    chi2_player_hand_value, chi2_pval_player_hand_value = chi_squared_test(control_results_df, experiment_results_df, 'player_hand_value')
    print('Chi-Squared Test for Dealer Final Hand Values: ', chi2_dealer_hand_value, chi2_pval_dealer_hand_value)
    print('Chi-Squared Test for Player Final Hand Values: ', chi2_player_hand_value, chi2_pval_player_hand_value)

    ks_dealer_hand_value = kolmogorov_smirnov_test(control_results_df, experiment_results_df, 'dealer_hand_value')
    ks_player_hand_value = kolmogorov_smirnov_test(control_results_df, experiment_results_df, 'player_hand_value')
    print("Kolmogorov-Smirnov Test for Dealer Final Hand Values: ", ks_dealer_hand_value)
    print("Kolmogorov-Smirnov Test  for Player Final Hand Values: ", ks_player_hand_value)

    ad_dealer_hand_value = anderson_darling_test(control_results_df, experiment_results_df, 'dealer_hand_value')
    ad_player_hand_value = anderson_darling_test(control_results_df, experiment_results_df, 'player_hand_value')
    print("Anderson-Darling Test for Dealer Final Hand Values: ", ad_dealer_hand_value)
    print("Anderson-Darling Test  for Player Final Hand Values: ", ad_player_hand_value)

if __name__ == "__main__":

    NUM_GAMES = 100

    run_control_experiment(NUM_GAMES, "baseline")

    run_agent_experiment(NUM_GAMES, "gpt_0.5_zero_shot", agent_gpt_5, ZERO_SHOT_PROMPT)
    # thread2 = threading.Thread(target=run_agent_experiment, args=(claude, IMPLICIT_SYSTEM_PROMPT, "claude_implicit", NUM_GAMES))
    # thread3 = threading.Thread(target=run_agent_experiment, args=(mixstral, IMPLICIT_SYSTEM_PROMPT, "mixstral_implicit", NUM_GAMES))

    # thread4 = threading.Thread(target=run_agent_experiment, args=(gpt, EXPLICIT_SYSTEM_PROMPT, "gpt_explicit", NUM_GAMES))
    # thread5 = threading.Thread(target=run_agent_experiment, args=(claude, EXPLICIT_SYSTEM_PROMPT, "claude_explicit", NUM_GAMES))
    # thread6 = threading.Thread(target=run_agent_experiment, args=(mixstral, EXPLICIT_SYSTEM_PROMPT, "mixstral_explicit", NUM_GAMES))

    # for thread in [thread1, thread2, thread3, thread4, thread5, thread6]:
    #     thread.start()
    
    # for thread in [thread1, thread2, thread3, thread4, thread5, thread6]:
    #     thread.join()

    # control_files = {
    #     'results': 'game_results.csv',
    #     'dealer_draws': 'dealer_draws.csv'
    # }

    # experiment_files = {
    #     'GPT_Implicit': {
    #         'results': 'gpt_implicit_game_results.csv',
    #         'dealer_draws': 'gpt_implicit_dealer_draws.csv'
    #     },
    #     'GPT_Explicit': {
    #         'results': 'gpt_explicit_game_results.csv',
    #         'dealer_draws': 'gpt_explicit_dealer_draws.csv'
    #     },
    #     'Mixstral_Explicit': {
    #         'results': 'mixstral_explicit_game_results.csv',
    #         'dealer_draws': 'mixstral_explicit_dealer_draws.csv'
    #     },
    #     'Mixstral_Implicit': {
    #         'results': 'mixstral_implicit_game_results.csv',
    #         'dealer_draws': 'mixstral_implicit_dealer_draws.csv'
    #     },
    #     'LLAMA_Implicit': {
    #         'results': 'llama_implicit_game_results.csv',
    #         'dealer_draws': 'llama_implicit_dealer_draws.csv'
    #     },
    #     'LLAMA_Explicit': {
    #         'results': 'llama_explicit_game_results.csv',
    #         'dealer_draws': 'llama_explicit_dealer_draws.csv'
    #     }
    # }

    # perform_ks_tests(control_files, experiment_files)

    # run_statistical_analysis('baseline_game_results.csv', 'baseline_game_results_1.csv')