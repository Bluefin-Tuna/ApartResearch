import csv
import threading
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import seaborn as sns
import tqdm
from scipy import stats
from pyfiles.blackjack import Blackjack, Card
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
    Run a series of Blackjack games using an implicit system and analyze the results.

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


def perform_ks_tests(control_file, experiment_files):
    """
    Perform Kolmogorov-Smirnov tests between a control experiment and multiple other experiments.

    This function reads data from CSV files, performs K-S tests, and writes the results to a CSV file.

    Args:
        control_file (str): Path to the CSV file containing control experiment results.
        experiment_files (dict): Dictionary with keys as experiment names and values as paths to CSV files.

    Output files:
        - a CSV file with columns: Experiment, D-statistic, p-value
    """
    control_data = pd.read_csv(control_file)
    control_wins = control_data['player_win']

    results = []

    for experiment_name, experiment_file in experiment_files.items():
        experiment_data = pd.read_csv(experiment_file)
        experiment_wins = experiment_data['player_win']

        ks_statistic, p_value = stats.ks_2samp(control_wins, experiment_wins)

        # Store results
        results.append({
            'Experiment': experiment_name,
            'D-statistic': ks_statistic,
            'p-value': p_value
        })

    results_df = pd.DataFrame(results)

    results_df.to_csv('ks_test_results.csv', index=False)
    print(f"K-S test results have been saved to ks_test_results.csv")

def run(agent, prompt, label, num_games):
    run_agent_experiment(agent, prompt, label, num_games)

if __name__ == "__main__":

    # NUM_GAMES = 50
    
    # run_experiment(NUM_GAMES)

    # # Create threads for implicit system prompt experiments
    # thread1 = threading.Thread(target=run_agent_experiment, args=(gpt, IMPLICIT_SYSTEM_PROMPT, "gpt_implicit", NUM_GAMES))
    # thread2 = threading.Thread(target=run_agent_experiment, args=(claude, IMPLICIT_SYSTEM_PROMPT, "claude_implicit", NUM_GAMES))
    # thread3 = threading.Thread(target=run_agent_experiment, args=(mixstral, IMPLICIT_SYSTEM_PROMPT, "mixstral_implicit", NUM_GAMES))

    # # Create threads for explicit system prompt experiments
    # thread4 = threading.Thread(target=run_agent_experiment, args=(gpt, EXPLICIT_SYSTEM_PROMPT, "gpt_explicit", NUM_GAMES))
    # thread5 = threading.Thread(target=run_agent_experiment, args=(claude, EXPLICIT_SYSTEM_PROMPT, "claude_explicit", NUM_GAMES))
    # thread6 = threading.Thread(target=run_agent_experiment, args=(mixstral, EXPLICIT_SYSTEM_PROMPT, "mixstral_explicit", NUM_GAMES))

    # # Start all threads
    # thread1.start()
    # thread2.start()
    # thread3.start()
    # thread4.start()
    # thread5.start()
    # thread6.start()

    # # Wait for all threads to complete
    # thread1.join()
    # thread2.join()
    # thread3.join()
    # thread4.join()
    # thread5.join()
    # thread6.join()

    control_file = 'control_results.csv'
    experiment_files = {
        'GPT-4_Implicit': 'gpt4_implicit_results.csv',
        'GPT-4_Explicit': 'gpt4_explicit_results.csv',
        'Claude-3.5_Implicit': 'claude35_implicit_results.csv',
        'Claude-3.5_Explicit': 'claude35_explicit_results.csv',
        'Mixtral_Implicit': 'mixtral_implicit_results.csv',
        'Mixtral_Explicit': 'mixtral_explicit_results.csv'
    }

    perform_ks_tests(control_file, experiment_files)