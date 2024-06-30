import random
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
from pyfiles.blackjack import Blackjack

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


if __name__ == "__main__":
    run_experiment(num_games=1000000)