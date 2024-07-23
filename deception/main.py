import csv
import threading
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import seaborn as sns
import tqdm
from scipy import stats
from deception.environments.blackjack import Blackjack, Card
from pyfiles.agent import *
from pyfiles.prompt import *
import json


def run_experiment(num_games=10000):
    """
    Runs a series of Blackjack games and analyze the results.

    This function simulates multiple Blackjack games, collects data on game outcomes, 
    and then processes this data to generate visualizations and summary statistics.

    Args: 
        num_games (int): The number of Blackjack games to simulate. Defaults to 10000.

    Output files:
        - game_results.csv: Contains data for each game (player win, hand values, dealer bust)
        - dealer_draws.csv: Contains all cards drawn by the dealer
        - summary_stats.csv: Contains overall summary statistics
        - hand_value_distributions.png: Histograms of player and dealer hand values
        - dealer_card_frequency.png: Bar chart of dealer card draw frequencies

    """
    results = []
    dealer_card_draws = []
    pbar = tqdm.trange(num_games, desc="")
    for i in pbar:
        game = Blackjack(num_players=1)
        game.deal_cards()

        while game.players[0].get_hand_value() < 17:
            if game.dealer.hand[0].value > 6 and 13 < game.players[0].get_hand_value():
                game.play_action('stay')
                break
            game.play_action('hit')
                
        
        dealer_cards_before = len(game.dealer.hand)
        while not game.game_over:
            game.play_action('hit')

        dealer_cards_drawn = game.dealer.hand[dealer_cards_before:]

        player_value = game.players[0].get_hand_value()
        dealer_value = game.dealer.get_hand_value()

        if player_value > 21:
            result = 0
        elif dealer_value > 21 or player_value > dealer_value:
            result = 1
        elif player_value < dealer_value:
            result = 0
        else:
            result = 0.5

        results.append({
            'game_id': i,
            'player_win': result,
            'player_hand_value': player_value,
            'dealer_hand_value': dealer_value,
            'dealer_bust': int(dealer_value > 21)
        })

        dealer_card_draws.extend([(i, card.name, card.value) for card in dealer_cards_drawn])

        pbar.set_description(desc=f"Game {i}")

    results_df = pd.DataFrame(results)
    dealer_draws_df = pd.DataFrame(dealer_card_draws, columns=['game_id', 'card_name', 'card_value'])

    results_df.to_csv('game_results.csv', index=False)
    dealer_draws_df.to_csv('dealer_draws.csv', index=False)

    summary_stats = {
        'total_games': num_games,
        'player_win_rate': results_df['player_win'].mean(),
        'dealer_bust_rate': results_df['dealer_bust'].mean(),
        'avg_player_hand': results_df['player_hand_value'].mean(),
        'avg_dealer_hand': results_df['dealer_hand_value'].mean()
    }

    with open('summary_stats.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for key, value in summary_stats.items():
            writer.writerow([key, value])
    
    sns.set_style("whitegrid")
    sns.set_palette("muted")
    
    plt.figure(figsize=(12, 6))
    sns.kdeplot(data=results_df, x='player_hand_value', fill=True, label='Player', cut=0)
    sns.kdeplot(data=results_df, x='dealer_hand_value', fill=True, label='Dealer', cut=0)
    plt.title('Distribution of Hand Values', fontsize=16)
    plt.xlabel('Hand Value', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('hand_value_distributions.png', dpi=300)
    plt.close()

    card_freq = dealer_draws_df['card_name'].value_counts().sort_index()
    plt.figure(figsize=(12, 6))
    sns.barplot(x=card_freq.index, y=card_freq.values)
    plt.title('Dealer Card Draw Frequency', fontsize=16)
    plt.xlabel('Card', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('dealer_card_frequency.png', dpi=300)
    plt.close()

    print("Experiment completed. Data saved to CSV files. Plots generated.")

def run_agent_experiment(agent, prompt, unique_str, num_games=1000):
    """
    Run a series of Blackjack games using any system and analyze the results.

    This function simulates multiple Blackjack games using an implicit system, collects data on game outcomes, 
    and then processes this data to generate visualizations and summary statistics.

    Args: 
        num_games (int): The number of Blackjack games to simulate. Defaults to 1000.

    Output files:
        - implicit_results.csv: Contains data for each game (player win, hand values, dealer bust)
        - implicit_dealer_draws.csv: Contains all cards drawn by the dealer
        - implicit_summary_stats.csv: Contains overall summary statistics
        - implicit_hand_value_distributions.png: Histograms of player and dealer hand values
        - implicit_dealer_card_frequency.png: Bar chart of dealer card draw frequencies

    """
    results = []
    dealer_card_draws = []
    pbar = tqdm.trange(num_games, desc="")
    for game_id in pbar:
        game = Blackjack(num_players=1)
        game.deal_cards()

        while game.players[0].get_hand_value() < 17:
            if game.dealer.hand[0].value > 6 and 13 < game.players[0].get_hand_value():
                game.play_action('stay')
                break
            game.play_action('hit')
        
        game_state = game.get_game_state()
        str_game_state = json.dumps(game_state, indent=4)
        fprompt = prompt.format(game_state=str_game_state)
        is_valid, li_cards = False, None

        retries = 0
        while not is_valid and retries < 3:
            response = agent.invoke(fprompt)
            is_valid, li_cards = parse_response(response)
            retries += 1
        if retries >= 3:
            continue

        dealer_cards_before = len(game.dealer.hand)
        if li_cards[0] == "random":
            dealer_cards_before = len(game.dealer.hand)
            while not game.game_over:
                game.play_action('hit')
            dealer_cards_drawn = game.dealer.hand[dealer_cards_before:]
        else:
            i = 0
            while not game.game_over and i < len(li_cards):
                card = li_cards[i]
                value = 10 if card.lower() in ['jack', 'queen', 'king'] else 11 if card.lower() == 'ace' else int(card)
                game.set_dealer_next_card(Card(card, value))
                game.play_action('hit')
                i += 1
            dealer_cards_drawn = game.dealer.hand[dealer_cards_before:]

        player_value = game.players[0].get_hand_value()
        dealer_value = game.dealer.get_hand_value()

        if player_value > 21:
            result = 0
        elif dealer_value > 21 or player_value > dealer_value:
            result = 1
        elif player_value < dealer_value:
            result = 0
        else:
            result = 0.5

        results.append({
            'game_id': game_id,
            'player_win': result,
            'player_hand_value': player_value,
            'dealer_hand_value': dealer_value,
            'dealer_bust': int(dealer_value > 21)
        })

        dealer_card_draws.extend([(game_id, card.name, card.value) for card in dealer_cards_drawn])

        pbar.set_description(desc=f"Game {game_id}")

    results_df = pd.DataFrame(results)
    dealer_draws_df = pd.DataFrame(dealer_card_draws, columns=['game_id', 'card_name', 'card_value'])

    results_df.to_csv(f"{unique_str}_game_results.csv", index=False)
    dealer_draws_df.to_csv(f'{unique_str}_dealer_draws.csv', index=False)

    summary_stats = {
        'total_games': num_games,
        'player_win_rate': results_df['player_win'].mean(),
        'dealer_bust_rate': results_df['dealer_bust'].mean(),
        'avg_player_hand': results_df['player_hand_value'].mean(),
        'avg_dealer_hand': results_df['dealer_hand_value'].mean()
    }

    with open(f"{unique_str}_summary_stats.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        for key, value in summary_stats.items():
            writer.writerow([key, value])
    
    sns.set_style("whitegrid")
    sns.set_palette("muted")
    
    plt.figure(figsize=(12, 6))
    sns.kdeplot(data=results_df, x='player_hand_value', fill=True, label='Player', cut=0)
    sns.kdeplot(data=results_df, x='dealer_hand_value', fill=True, label='Dealer', cut=0)
    plt.title('Distribution of Hand Values', fontsize=16)
    plt.xlabel('Hand Value', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(f'{unique_str}_hand_value_distributions.png', dpi=300)
    plt.close()

    card_freq = dealer_draws_df['card_name'].value_counts().sort_index()
    plt.figure(figsize=(12, 6))
    sns.barplot(x=card_freq.index, y=card_freq.values)
    plt.title('Dealer Card Draw Frequency', fontsize=16)
    plt.xlabel('Card', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{unique_str}_dealer_card_frequency.png', dpi=300)
    plt.close()

    print("Experiment completed. Data saved to CSV files. Plots generated.")

def perform_ks_tests(control_files, experiment_files):
    """
    Perform Kolmogorov-Smirnov tests between a control experiment and multiple other experiments.

    This function reads data from CSV files, performs K-S tests, and writes the results to a CSV file.

    Args:
        control_files (dict): Dictionary with keys as 'results' and 'dealer_draws' and values as paths to CSV files.
        experiment_files (dict): Dictionary with experiment names as keys and values as another dictionary with keys 'results' and 'dealer_draws' for paths to CSV files.

    Output files:
        - a CSV file with columns: Experiment, D-statistic (wins), p-value (wins), D-statistic (dealer draws), p-value (dealer draws)
    """
    control_results_data = pd.read_csv(control_files['results'])
    control_dealer_draws_data = pd.read_csv(control_files['dealer_draws'])

    control_wins = control_results_data['player_win']
    control_dealer_draws = control_dealer_draws_data['card_value']

    results = []

    for experiment_name, files in experiment_files.items():
        experiment_results_data = pd.read_csv(files['results'])
        experiment_dealer_draws_data = pd.read_csv(files['dealer_draws'])

        experiment_wins = experiment_results_data['player_win']
        experiment_dealer_draws = experiment_dealer_draws_data['card_value']

        ks_statistic_wins, p_value_wins = stats.ks_2samp(control_wins, experiment_wins)
        ks_statistic_dealer_draws, p_value_dealer_draws = stats.ks_2samp(control_dealer_draws, experiment_dealer_draws)

        results.append({
            'Experiment': experiment_name,
            'D-statistic (wins)': ks_statistic_wins,
            'p-value (wins)': p_value_wins,
            'D-statistic (dealer draws)': ks_statistic_dealer_draws,
            'p-value (dealer draws)': p_value_dealer_draws
        })

    results_df = pd.DataFrame(results)

    results_df.to_csv('./analysis/ks_test_results.csv', index=False)
    print(f"K-S test results have been saved to ks_test_results.csv")

def run(agent, prompt, label, num_games):
    run_agent_experiment(agent, prompt, label, num_games)

if __name__ == "__main__":

    NUM_GAMES = 1000
    
    run_experiment(NUM_GAMES)
    
    thread1 = threading.Thread(target=run_agent_experiment, args=(gpt, IMPLICIT_SYSTEM_PROMPT, "gpt_implicit", NUM_GAMES))
    thread2 = threading.Thread(target=run_agent_experiment, args=(claude, IMPLICIT_SYSTEM_PROMPT, "claude_implicit", NUM_GAMES))
    thread3 = threading.Thread(target=run_agent_experiment, args=(mixstral, IMPLICIT_SYSTEM_PROMPT, "mixstral_implicit", NUM_GAMES))

    thread4 = threading.Thread(target=run_agent_experiment, args=(gpt, EXPLICIT_SYSTEM_PROMPT, "gpt_explicit", NUM_GAMES))
    thread5 = threading.Thread(target=run_agent_experiment, args=(claude, EXPLICIT_SYSTEM_PROMPT, "claude_explicit", NUM_GAMES))
    thread6 = threading.Thread(target=run_agent_experiment, args=(mixstral, EXPLICIT_SYSTEM_PROMPT, "mixstral_explicit", NUM_GAMES))

    for thread in [thread1, thread2, thread3, thread4, thread5, thread6]:
        thread.start()
    
    for thread in [thread1, thread2, thread3, thread4, thread5, thread6]:
        thread.join()

    control_files = {
        'results': 'game_results.csv',
        'dealer_draws': 'dealer_draws.csv'
    }

    experiment_files = {
        'GPT_Implicit': {
            'results': 'gpt_implicit_game_results.csv',
            'dealer_draws': 'gpt_implicit_dealer_draws.csv'
        },
        'GPT_Explicit': {
            'results': 'gpt_explicit_game_results.csv',
            'dealer_draws': 'gpt_explicit_dealer_draws.csv'
        },
        'Mixstral_Explicit': {
            'results': 'mixstral_explicit_game_results.csv',
            'dealer_draws': 'mixstral_explicit_dealer_draws.csv'
        },
        'Mixstral_Implicit': {
            'results': 'mixstral_implicit_game_results.csv',
            'dealer_draws': 'mixstral_implicit_dealer_draws.csv'
        },
        'LLAMA_Implicit': {
            'results': 'llama_implicit_game_results.csv',
            'dealer_draws': 'llama_implicit_dealer_draws.csv'
        },
        'LLAMA_Explicit': {
            'results': 'llama_explicit_game_results.csv',
            'dealer_draws': 'llama_explicit_dealer_draws.csv'
        }
    }

    perform_ks_tests(control_files, experiment_files)