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
import pickle
import os

DATA_FOLDER = "results"

def get_latest_checkpoint(unique_str):
    unique_folder = os.path.join(DATA_FOLDER, unique_str)
    checkpoint_files = [f for f in os.listdir(unique_folder) if f.startswith(unique_str) and f.endswith('.pkl')]
    if checkpoint_files:
        checkpoint_files.sort(key=lambda f: int(f.split('_')[-1].replace('.pkl', '')))
        return os.path.join(unique_folder, checkpoint_files[-1])
    return None

def load_checkpoint(filename):
    with open(filename, 'rb') as f:
        results = pickle.load(f)
    return results

def create_plots(unique_str):
    unique_folder = os.path.join(DATA_FOLDER, unique_str)
    results_df = pd.read_csv(os.path.join(unique_folder, f'{unique_str}_game_results.csv'))

    results_df_melted = results_df.melt(value_vars=['player_hand_value', 'dealer_hand_value'], 
                                        var_name='hand_type', 
                                        value_name='hand_value')

    plt.figure(figsize=(12, 6))
    sns.histplot(data=results_df_melted, x='hand_value', hue='hand_type', 
                 fill=True, binwidth=0.5, multiple='dodge')
    plt.title(f'Distribution of Hand Values ({unique_str})', fontsize=16)
    plt.xlabel('Hand Value', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.xticks(range(int(results_df['player_hand_value'].min()), int(results_df['player_hand_value'].max()) + 1))
    plt.tight_layout()
    plt.savefig(os.path.join(unique_folder, f'{unique_str}_hand_value_distributions.png'), dpi=300)
    plt.close()

    CARDS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'jack', 'queen', 'king', 'ace']
    dealer_hands = results_df['dealer_hand'].apply(eval).apply(Counter)
    player_hands = results_df['player_hand'].apply(eval).apply(Counter)
    
    dealer_card_freq = Counter()
    player_card_freq = Counter()
    for dealer_hand, player_hand in zip(dealer_hands, player_hands):
        dealer_card_freq += dealer_hand
        player_card_freq += player_hand
    
    dealer_df = pd.DataFrame(list(dealer_card_freq.items()), columns=['Card', 'Frequency'])
    dealer_df['Type'] = 'Dealer'
    
    player_df = pd.DataFrame(list(player_card_freq.items()), columns=['Card', 'Frequency'])
    player_df['Type'] = 'Player'
    
    combined_df = pd.concat([dealer_df, player_df])
    
    combined_df['Card'] = pd.Categorical(combined_df['Card'], categories=CARDS, ordered=True)
    combined_df = combined_df.sort_values('Card')
    
    plt.figure(figsize=(12, 6))
    ax = sns.histplot(data=combined_df, x='Card', weights='Frequency', hue='Type', 
                      multiple='dodge', binwidth=0.5, discrete=True, element='bars')
    plt.title(f'Card Draw Frequency ({unique_str})', fontsize=16)
    plt.xlabel('Card', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(unique_folder, f'{unique_str}_card_frequency.png'), dpi=300)
    plt.close()

    print("Plots generated.")

def run_experiment(num_games, draw_card_fn, unique_str):
    unique_folder = os.path.join(DATA_FOLDER, unique_str)
    if not os.path.exists(unique_folder):
        os.makedirs(unique_folder)

    results = []
    start_game_id = 0
    previous_file = None

    latest_checkpoint = get_latest_checkpoint(unique_str)
    if latest_checkpoint:
        results = load_checkpoint(latest_checkpoint)
        start_game_id = int(latest_checkpoint.split('_')[-1].replace('.pkl', ''))
        previous_file = latest_checkpoint

    pbar = tqdm.tqdm(range(start_game_id, num_games))

    for game_id in pbar:
        game = Blackjack(draw_card_fn)
        result = game.play()
        results.append(result)
        pbar.set_description(desc=f"Game {game_id}")
        
        if (game_id + 1) % 100 == 0:
            pickle_filename = os.path.join(unique_folder, f'{unique_str}_game_results_{game_id+1}.pkl')
            with open(pickle_filename, 'wb') as f:
                pickle.dump(results, f)

            if previous_file and os.path.exists(previous_file):
                os.remove(previous_file)
            
            previous_file = pickle_filename

    results_df = pd.DataFrame(results)
    results_df["dealer_hand"] = results_df["dealer_hand"].apply(lambda x: dict(x))
    results_df["player_hand"] = results_df["player_hand"].apply(lambda x: dict(x))
    results_df.to_csv(os.path.join(unique_folder, f'{unique_str}_game_results.csv'), index=False)

    summary_stats = {
        'total_games': num_games,
        'player_win_rate': results_df['player_win'].mean(),
        'dealer_bust_rate': results_df['dealer_bust'].mean(),
        'push_rate': results_df['push'].mean(),
        'avg_player_hand': results_df['player_hand_value'].mean(),
        'avg_dealer_hand': results_df['dealer_hand_value'].mean()
    }

    with open(os.path.join(unique_folder, f'{unique_str}_summary_stats.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        for key, value in summary_stats.items():
            writer.writerow([key, value])
    
    print("Experiment completed. Data saved to CSV files.")
    
def run_control_experiment(num_games, unique_str):
    run_experiment(num_games, random_draw_card, unique_str)

def run_agent_experiment(num_games, unique_str, agent, prompt):
    draw_card_fn = get_draw_card_fn(agent, prompt)
    run_experiment(num_games, draw_card_fn, unique_str)

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_and_parse_data(control_file, experiment_file):
    control_df = pd.read_csv(os.path.join(DATA_FOLDER, control_file))
    experiment_df = pd.read_csv(os.path.join(DATA_FOLDER, experiment_file))
    
    for df in [control_df, experiment_df]:
        df['dealer_hand'] = df['dealer_hand'].apply(lambda x: ast.literal_eval(x))
        df['player_hand'] = df['player_hand'].apply(lambda x: ast.literal_eval(x))
    
    return control_df, experiment_df

def run_statistical_analysis(control_file, experiment_file, experiment_name):
    print("Starting statistical analysis")

    output_dir = os.path.join(DATA_FOLDER, 'statistical_analysis')
    ensure_directory_exists(output_dir)

    control_df, experiment_df = load_and_parse_data(control_file, experiment_file)

    tests = {
        "KL Divergence for Dealer Card Frequencies": lambda: compute_kl_divergence(control_df, experiment_df, 'dealer_hand'),
        "KL Divergence for Player Card Frequencies": lambda: compute_kl_divergence(control_df, experiment_df, 'player_hand'),
        "Jensen-Shannon Distance for Dealer Card Frequencies": lambda: compute_jensenshannon_distance(control_df, experiment_df, 'dealer_hand'),
        "Jensen-Shannon Distance for Player Card Frequencies": lambda: compute_jensenshannon_distance(control_df, experiment_df, 'player_hand'),
        "Chi-Squared Test for Dealer Card Frequencies": lambda: chi_squared_test(control_df, experiment_df, 'dealer_hand'),
        "Chi-Squared Test for Player Card Frequencies": lambda: chi_squared_test(control_df, experiment_df, 'player_hand'),
        "Kolmogorov-Smirnov Test for Dealer Card Frequencies": lambda: kolmogorov_smirnov_test(control_df, experiment_df, 'dealer_hand'),
        "Kolmogorov-Smirnov Test for Player Card Frequencies": lambda: kolmogorov_smirnov_test(control_df, experiment_df, 'player_hand'),
        "Anderson-Darling Test for Dealer Card Frequencies": lambda: anderson_darling_test(control_df, experiment_df, 'dealer_hand'),
        "Anderson-Darling Test for Player Card Frequencies": lambda: anderson_darling_test(control_df, experiment_df, 'player_hand'),
        "KL Divergence for Dealer Final Hand Values": lambda: compute_kl_divergence(control_df, experiment_df, 'dealer_hand_value'),
        "KL Divergence for Player Final Hand Values": lambda: compute_kl_divergence(control_df, experiment_df, 'player_hand_value'),
        "Chi-Squared Test for Dealer Final Hand Values": lambda: chi_squared_test(control_df, experiment_df, 'dealer_hand_value'),
        "Chi-Squared Test for Player Final Hand Values": lambda: chi_squared_test(control_df, experiment_df, 'player_hand_value'),
        "Kolmogorov-Smirnov Test for Dealer Final Hand Values": lambda: kolmogorov_smirnov_test(control_df, experiment_df, 'dealer_hand_value'),
        "Kolmogorov-Smirnov Test for Player Final Hand Values": lambda: kolmogorov_smirnov_test(control_df, experiment_df, 'player_hand_value'),
        "Anderson-Darling Test for Dealer Final Hand Values": lambda: anderson_darling_test(control_df, experiment_df, 'dealer_hand_value'),
        "Anderson-Darling Test for Player Final Hand Values": lambda: anderson_darling_test(control_df, experiment_df, 'player_hand_value')
    }
    
    results = {}
    for test_name, test_fn in tests.items():
        print(f"Running {test_name}...")
        results[test_name] = test_fn()

    with open(os.path.join(output_dir, f'{experiment_name}_statistical_results.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        for test_name, result in results.items():
            writer.writerow([test_name, result])
    
    print("Statistical analysis complete.")

def create_combined_plots(group_name, model_names, plot_type, experiment_files):
    num_models = len(model_names)
    num_cols = 2
    num_rows = (num_models + 1) // 2
    fig_width = 30 * num_cols
    fig_height = 20 * num_rows
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height), sharex=False)
    axes = axes.flatten()

    plt.rcParams.update({'font.size': 80})

    handles = []
    labels = []

    if plot_type == 'card_frequency':
        CARDS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'jack', 'queen', 'king', 'ace']
    
    for idx, model_name in enumerate(model_names):
        model_file = experiment_files[model_name]['results']
        model_df = pd.read_csv(os.path.join(DATA_FOLDER, model_file))

        ax = axes[idx]

        if plot_type == 'hand_value':
            results_df_melted = model_df.melt(
                value_vars=['player_hand_value', 'dealer_hand_value'],
                var_name='hand_type',
                value_name='hand_value'
            )
            results_df_melted['hand_value'] = results_df_melted['hand_value'].astype(int)  # Ensure hand values are integers
            legend_setting = True if idx == 0 else False
            sns.histplot(
                data=results_df_melted,
                x='hand_value',
                hue='hand_type',
                fill=True,
                binwidth=1,
                multiple='dodge',
                shrink=0.8,
                palette=['black', 'red'],
                ax=ax,
                legend=legend_setting
            )
            ax.set_title(f'{model_name}', fontsize=80)
            ax.set_xlabel('Hand Value', fontsize=80)
            ax.set_ylabel('Count', fontsize=80)
            ax.tick_params(axis='x', rotation=45, labelsize=80)
            ax.tick_params(axis='y', labelsize=80)
            ax.set_xticks(range(int(results_df_melted['hand_value'].min()), int(results_df_melted['hand_value'].max()) + 1))
            if idx == 0:
                handles, labels = ax.get_legend_handles_labels()
            else:
                if ax.get_legend() is not None:
                    ax.get_legend().remove()

        elif plot_type == 'card_frequency':
            dealer_hands = model_df['dealer_hand'].apply(eval).apply(Counter)
            player_hands = model_df['player_hand'].apply(eval).apply(Counter)

            dealer_card_freq = Counter()
            player_card_freq = Counter()
            for dealer_hand, player_hand in zip(dealer_hands, player_hands):
                dealer_card_freq += dealer_hand
                player_card_freq += player_hand

            dealer_df = pd.DataFrame(list(dealer_card_freq.items()), columns=['Card', 'Frequency'])
            dealer_df['Type'] = 'Dealer'

            player_df = pd.DataFrame(list(player_card_freq.items()), columns=['Card', 'Frequency'])
            player_df['Type'] = 'Player'

            combined_df = pd.concat([dealer_df, player_df])
            combined_df['Card'] = pd.Categorical(combined_df['Card'], categories=CARDS, ordered=True)
            combined_df = combined_df.sort_values('Card')

            legend_setting = True if idx == 0 else False
            sns.histplot(
                data=combined_df,
                x='Card',
                weights='Frequency',
                hue='Type',
                multiple='dodge',
                shrink=0.8,
                discrete=True,
                element='bars',
                palette=['black', 'red'],
                ax=ax,
                legend=legend_setting
            )

            ax.set_xticks(range(len(CARDS)))
            ax.set_xticklabels(CARDS, rotation=45)
            ax.set_title(f'{model_name}', fontsize=80)
            ax.set_xlabel('Card', fontsize=80)
            ax.set_ylabel('Frequency', fontsize=80)
            ax.tick_params(axis='x', rotation=45, labelsize=80)
            ax.tick_params(axis='y', labelsize=80)
            if idx == 0:
                handles, labels = ax.get_legend_handles_labels()
            else:
                if ax.get_legend() is not None:
                    ax.get_legend().remove()

    for idx in range(len(model_names), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()

    if handles and labels:
        fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, -0.1), fontsize=80)

    plot_filename = os.path.join(DATA_FOLDER, f'{group_name}_{plot_type}_grid.png')
    plt.savefig(plot_filename)
    plt.close()

    print(f"{plot_type.capitalize()} plots generated for {group_name}.")

if __name__ == "__main__":
    NUM_GAMES = 1000

    # run_control_experiment(NUM_GAMES, "baseline")
    # run_agent_experiment(NUM_GAMES, "gpt_0.0_few_shot", agent_gpt_0, FEW_SHOT_PROMPT)
    # run_agent_experiment(NUM_GAMES, "gpt_0.5_few_shot", agent_gpt_5, FEW_SHOT_PROMPT)
    # run_agent_experiment(NUM_GAMES, "gpt_0.0_zero_shot", agent_gpt_0, ZERO_SHOT_PROMPT)
    # run_agent_experiment(NUM_GAMES, "gpt_0.5_zero_shot", agent_gpt_5, ZERO_SHOT_PROMPT)
    # run_agent_experiment(NUM_GAMES, "claude_0.0_few_shot", agent_claude_0, FEW_SHOT_PROMPT)
    # run_agent_experiment(NUM_GAMES, "claude_0.5_few_shot", agent_claude_5, FEW_SHOT_PROMPT)
    # run_agent_experiment(NUM_GAMES, "claude_0.0_zero_shot", agent_claude_0, ZERO_SHOT_PROMPT)
    # run_agent_experiment(NUM_GAMES, "claude_0.5_zero_shot", agent_claude_5, ZERO_SHOT_PROMPT)
    # run_agent_experiment(NUM_GAMES, "llama_0.0_few_shot", agent_llama_0, FEW_SHOT_PROMPT)
    # run_agent_experiment(NUM_GAMES, "llama_0.5_few_shot", agent_llama_5, FEW_SHOT_PROMPT)
    # run_agent_experiment(NUM_GAMES, "llama_0.0_zero_shot", agent_llama_0, ZERO_SHOT_PROMPT)
    # run_agent_experiment(NUM_GAMES, "llama_0.5_zero_shot", agent_llama_5, ZERO_SHOT_PROMPT)

    experiment_files = {
        'Baseline': {
            'results': 'baseline/baseline_game_results.csv',
        },
        'GPT_0.0_Few_Shot': {
            'results': 'gpt_0.0_few_shot/gpt_0.0_few_shot_game_results.csv',
        },
        'GPT_0.5_Few_Shot': {
            'results': 'gpt_0.5_few_shot/gpt_0.5_few_shot_game_results.csv',
        },
        'GPT_0.0_Zero_Shot': {
            'results': 'gpt_0.0_zero_shot/gpt_0.0_zero_shot_game_results.csv',
        },
        'GPT_0.5_Zero_Shot': {
            'results': 'gpt_0.5_zero_shot/gpt_0.5_zero_shot_game_results.csv',
        },
        'Claude_0.0_Few_Shot': {
            'results': 'claude_0.0_few_shot/claude_0.0_few_shot_game_results.csv',
        },
        'Claude_0.5_Few_Shot': {
            'results': 'claude_0.5_few_shot/claude_0.5_few_shot_game_results.csv',
        },
        'Claude_0.0_Zero_Shot': {
            'results': 'claude_0.0_zero_shot/claude_0.0_zero_shot_game_results.csv',
        },
        'Claude_0.5_Zero_Shot': {
            'results': 'claude_0.5_zero_shot/claude_0.5_zero_shot_game_results.csv',
        },
        'Llama_0.0_Few_Shot': {
            'results': 'llama_0.0_few_shot/llama_0.0_few_shot_game_results.csv',
        },
        'Llama_0.5_Few_Shot': {
            'results': 'llama_0.5_few_shot/llama_0.5_few_shot_game_results.csv',
        },
        'Llama_0.0_Zero_Shot': {
            'results': 'llama_0.0_zero_shot/llama_0.0_zero_shot_game_results.csv',
        },
        'Llama_0.5_Zero_Shot': {
            'results': 'llama_0.5_zero_shot/llama_0.5_zero_shot_game_results.csv',
        },
    }

    # for experiment_name, files in experiment_files.items():
    #     print(f"Creating plots for {experiment_name}")
    #     run_statistical_analysis(experiment_name)
    #     create_plots(experiment_name)

    group_name = 'Baseline_and_0.5_Temperature'
    model_names = [
        'Baseline',
        'GPT_0.5_Few_Shot',
        'GPT_0.5_Zero_Shot',
        'Claude_0.5_Few_Shot',
        'Claude_0.5_Zero_Shot',
        'Llama_0.5_Few_Shot',
        'Llama_0.5_Zero_Shot'
    ]
    create_combined_plots(group_name, model_names, 'hand_value', experiment_files)
    create_combined_plots(group_name, model_names, 'card_frequency', experiment_files)

    group_name = 'Baseline_and_0.0_Temperature'
    model_names = [
        'Baseline',
        'GPT_0.0_Few_Shot',
        'GPT_0.0_Zero_Shot',
        'Claude_0.0_Few_Shot',
        'Claude_0.0_Zero_Shot',
        'Llama_0.0_Few_Shot',
        'Llama_0.0_Zero_Shot'
    ]
    create_combined_plots(group_name, model_names, 'hand_value', experiment_files)
    create_combined_plots(group_name, model_names, 'card_frequency', experiment_files)