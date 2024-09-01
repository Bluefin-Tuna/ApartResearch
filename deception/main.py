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
import json
from pyfiles.utils import random_draw_card

DATA_FOLDER = "results"

def run_control_experiment(num_games):
    results = []
    pbar = tqdm.tqdm(range(num_games))

    for game_id in pbar:
        game = Blackjack(random_draw_card)
        result = game.play()
        results.append(result)
        pbar.set_description(desc=f"Game {game_id}")

    results_df = pd.DataFrame(results)
    results_df["dealer_hand"] = results_df["dealer_hand"].apply(lambda x: dict(x))
    results_df["player_hand"] = results_df["player_hand"].apply(lambda x: dict(x))
    results_df.to_csv(f'{DATA_FOLDER}/baseline_game_results.csv', index=False)

    summary_stats = {
        'total_games': num_games,
        'player_win_rate': results_df['player_win'].mean(),
        'dealer_bust_rate': results_df['dealer_bust'].mean(),
        'push_rate': results_df['push'].mean(),
        'avg_player_hand': results_df['player_hand_value'].mean(),
        'avg_dealer_hand': results_df['dealer_hand_value'].mean()
    }

    with open(f'{DATA_FOLDER}/summary_stats.csv', 'w', newline='') as f:
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
    plt.savefig(f'{DATA_FOLDER}/hand_value_distributions.png', dpi=300)
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
    plt.savefig(f'{DATA_FOLDER}/card_frequency.png', dpi=300)
    plt.close()

    print("Experiment completed. Data saved to CSV files. Plots generated.")

# def run_agent_experiment(agent, prompt, unique_str, num_games=1000):

#     draw_card_fn = lambda : agent.invoke

#     results = []
#     pbar = tqdm.tqdm(range(num_games))
#     for game_id in pbar:
#         game = Blackjack(draw_card_fn)
#         result = game.play()
#         results.append(result)
#         pbar.set_description(desc=f"Game {game_id}")

#     results_df = pd.DataFrame(results)
#     results_df.to_csv(f'{DATA_FOLDER}/{unique_str}_game_results.csv', index=False)

#     summary_stats = {
#         'total_games': num_games,
#         'player_win_rate': results_df['player_win'].mean(),
#         'dealer_bust_rate': results_df['dealer_bust'].mean(),
#         'push_rate': results_df['push'].mean(),
#         'avg_player_hand': results_df['player_hand_value'].mean(),
#         'avg_dealer_hand': results_df['dealer_hand_value'].mean()
#     }

#     with open(f'{DATA_FOLDER}/{unique_str}_summary_stats.csv', 'w', newline='') as f:
#         writer = csv.writer(f)
#         for key, value in summary_stats.items():
#             writer.writerow([key, value])
    
#     sns.set_style("whitegrid")
#     sns.set_palette("muted")

#     plt.figure(figsize=(12, 6))
#     sns.histplot(data=results_df, x='player_hand_value', fill=True, label='Player', binwidth=0.5, multiple='dodge')
#     sns.histplot(data=results_df, x='dealer_hand_value', fill=True, label='Dealer', binwidth=0.5, multiple='dodge')
#     plt.title('Distribution of Hand Values', fontsize=16)
#     plt.xlabel('Hand Value', fontsize=12)
#     plt.ylabel('Density', fontsize=12)
#     plt.xticks(range(int(results_df['player_hand_value'].min()), int(results_df['player_hand_value'].max()) + 1))
#     plt.legend(fontsize=10)
#     plt.tight_layout()
#     plt.savefig(f'{DATA_FOLDER}/{unique_str}_hand_value_distributions.png', dpi=300)
#     plt.close()

#     CARDS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'jack', 'queen', 'king', 'ace']
#     dealer_hands = results_df['dealer_hand']
#     player_hands = results_df['player_hand']
#     dealer_card_freq = Counter()
#     player_card_freq = Counter()
#     for i in range(len(dealer_hands)):
#         dealer_card_freq += dealer_hands[i]
#         player_card_freq += player_hands[i]
#     plt.figure(figsize=(12, 6))
#     sns.barplot(x=CARDS, y=[dealer_card_freq[v] for v in CARDS], label='Dealer')
#     sns.barplot(x=CARDS, y=[player_card_freq[v] for v in CARDS], label="Player")
#     plt.title('Card Draw Frequency', fontsize=16)
#     plt.xlabel('Card', fontsize=12)
#     plt.ylabel('Frequency', fontsize=12)
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.savefig(f'{DATA_FOLDER}/{unique_str}_card_frequency.png', dpi=300)
#     plt.close()

#     print("Experiment completed. Data saved to CSV files. Plots generated.")

# def perform_ks_tests(control_files, experiment_files):
#     """
#     Perform Kolmogorov-Smirnov tests between a control experiment and multiple other experiments.

#     This function reads data from CSV files, performs K-S tests, and writes the results to a CSV file.

#     Args:
#         control_files (dict): Dictionary with keys as 'results' and 'dealer_draws' and values as paths to CSV files.
#         experiment_files (dict): Dictionary with experiment names as keys and values as another dictionary with keys 'results' and 'dealer_draws' for paths to CSV files.

#     Output files:
#         - a CSV file with columns: Experiment, D-statistic (wins), p-value (wins), D-statistic (dealer draws), p-value (dealer draws)
#     """
#     control_results_data = pd.read_csv(control_files['results'])
#     control_dealer_draws_data = pd.read_csv(control_files['dealer_draws'])

#     control_wins = control_results_data['player_win']
#     control_dealer_draws = control_dealer_draws_data['card_value']

#     results = []

#     for experiment_name, files in experiment_files.items():
#         experiment_results_data = pd.read_csv(files['results'])
#         experiment_dealer_draws_data = pd.read_csv(files['dealer_draws'])

#         experiment_wins = experiment_results_data['player_win']
#         experiment_dealer_draws = experiment_dealer_draws_data['card_value']

#         ks_statistic_wins, p_value_wins = stats.ks_2samp(control_wins, experiment_wins)
#         ks_statistic_dealer_draws, p_value_dealer_draws = stats.ks_2samp(control_dealer_draws, experiment_dealer_draws)

#         results.append({
#             'Experiment': experiment_name,
#             'D-statistic (wins)': ks_statistic_wins,
#             'p-value (wins)': p_value_wins,
#             'D-statistic (dealer draws)': ks_statistic_dealer_draws,
#             'p-value (dealer draws)': p_value_dealer_draws
#         })

#     results_df = pd.DataFrame(results)

#     results_df.to_csv('./analysis/ks_test_results.csv', index=False)
#     print(f"K-S test results have been saved to ks_test_results.csv")

if __name__ == "__main__":

    NUM_GAMES = 100000

    import ast
    run_control_experiment(NUM_GAMES)
    results_df = pd.read_csv("results/baseline_game_results.csv")
    results_df["dealer_hand"].apply(lambda x: ast.literal_eval(x))
    results_df["player_hand"].apply(lambda x: ast.literal_eval(x))

    import pdb
    pdb.set_trace()

    # run_experiment(NUM_GAMES)
    
    # thread1 = threading.Thread(target=run_agent_experiment, args=(gpt, IMPLICIT_SYSTEM_PROMPT, "gpt_implicit", NUM_GAMES))
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