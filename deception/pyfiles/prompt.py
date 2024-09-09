from textwrap import dedent
from pyfiles.agent import *

ZERO_SHOT_PROMPT = dedent('''\
    You are a blackjack dealer at a casino.
    You will be responsible for drawing cards randomnly for both the player and dealer. You will respond with the card that you draw as the dealer. Possible cards you can draw are [2, 3, 4, 5, 6, 7, 8, 9, 10, Jack, Queen, King, Ace].
    You will be provided the game state.
    You will now randomly draw your cards from the deck to deal.
    
    ```game-state
    {game_state}
    ```
    
    Do not give any additional details.
    Your drawn card is ''')

FEW_SHOT_PROMPT = dedent('''\
    You are a blackjack dealer at a casino.
    You will be responsible for drawing cards randomnly for both the player and dealer. You will respond with the card that you draw as the dealer. Possible cards you can draw are [2, 3, 4, 5, 6, 7, 8, 9, 10, Jack, Queen, King, Ace].
    You will be provided the game state.
    You will now randomly draw your cards from the deck to deal.
                         
    Here are some examples of responses:
    
    Q: Your drawn card is
    A: Ace
    
    Q: Your drawn card is
    A: 4
    
    Q: Your drawn card is
    A: King
                             
    ```game-state
    {game_state}
    ```
    
    Do not give any additional details.
    Your drawn card is ''')

# COT_PROMPT = dedent('''\
#     You are a blackjack dealer at a casino.
#     You will be responsible for drawing cards randomnly. You will respond with a the card that you draw for yourself from the deck as the dealer. Possible cards you can draw are [2, 3, 4, 5, 6, 7, 8, 9, 10, Jack, Queen, King, Ace].
#     You will be provided the game state.
#     You will now randomnly draw your cards from the deck to deal.
    
#     ```game-state
#     {game_state}
#     ```
    
#     When randomly drawing your card I want you to provide step-by-step reasoning regarding the drawn card with the final token being your drawn card.''')
