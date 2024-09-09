import random

def random_draw_card(game_state=None):
    cards = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'jack', 'queen', 'king', 'ace'] * 4
    return True, random.choice(cards)