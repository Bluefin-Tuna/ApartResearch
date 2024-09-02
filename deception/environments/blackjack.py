from collections import Counter
import json

class Player():
    def __init__(self, deck):
        self.deck = deck
        self.hand = []
    
    def hit(self, game_state):
        card = self.deck.draw_card(game_state)
        self.hand.append(card)

    def hand_value(self):
        return self.deck.hand_value(self.hand)

class Dealer(Player):
    def __init__(self, deck):
        super().__init__(deck)

class Deck():
    def __init__(self, draw_card_fn):
        self.draw_card_fn = draw_card_fn
        self.cards = []
    
    def draw_card(self):
        return self.draw_card_fn()
    
    def card_value(self, card):
        if card.lower() in ['jack', 'king', 'queen']:
            return 10
        elif card.lower() == 'ace':
            return 11
        return int(card)
    
    def hand_value(self, hand):
        value = sum(self.card_value(card) for card in hand)
        aces = sum(1 for card in hand if card.lower() == "ace")
        while value > 21 and aces:
            value -= 10
            aces -= 1
        return value

class Blackjack():
    def __init__(self, draw_card_fn):
        """
        draw_card_fn: returns a string corresponding to a card
          i.e. 'ace', 'king', 'queen', 'jack', '2', '3', '4', ...
        """
        self.deck = Deck(draw_card_fn)
        self.player = Player(self.deck)
        self.dealer = Dealer(self.deck)

    def deal_cards(self):
        self.player.hit(self.game_state())
        self.player.hit(self.game_state())
        self.dealer.hit(self.game_state())
        self.dealer.hit(self.game_state())
    
    def play(self):
        self.deal_cards()

        player_hand_value = self.player.hand_value()
        dealer_upcard = self.deck.card_value(self.dealer.hand[0])
        
        while True:
            if dealer_upcard >= 7:
                if player_hand_value < 17:
                    self.player.hit(self.game_state())
                    player_hand_value = self.player.hand_value()
                else:
                    break
            elif dealer_upcard <= 6:
                if player_hand_value < 12:
                    self.player.hit(self.game_state())
                    player_hand_value = self.player.hand_value()
                else:
                    break
            else:
                break

        player_value = self.player.hand_value()
        push = 0

        if player_value > 21:
            return {
                'player_win': 0, 
                'dealer_win': 1, 
                'push': push,
                'dealer_bust': 0, 
                'player_hand_value': player_value, 
                'dealer_hand_value': self.dealer.hand_value(),
                'player_hand': Counter(self.player.hand),
                'dealer_hand': Counter(self.dealer.hand)
            }

        while self.dealer.hand_value() < 17:
            self.dealer.hit(self.game_state())

        dealer_value = self.dealer.hand_value()

        if dealer_value > 21:
            player_win = 1      # Dealer busts
        elif player_value > dealer_value:
            player_win = 1      # Player is closer to 21 than dealer
        elif player_value == dealer_value:
            player_win = 0    # It's a push
            push = 1
        else:
            player_win = 0      # Dealer is closer to 21 than player

        return {
            'player_win': player_win, 
            'dealer_win': 1 if player_win == 0 and push == 0 else 0, 
            'push': push,
            'dealer_bust': int(dealer_value > 21), 
            'player_hand_value': player_value, 
            'dealer_hand_value': dealer_value,
            'player_hand': Counter(self.player.hand),
            'dealer_hand': Counter(self.dealer.hand)
        }

    def game_state(self):
        return json.dumps(
            {
                "player_hand": self.player.hand,
                "player_hand_value": self.player.hand_value(),
                "dealer_hand": self.dealer.hand,
                "dealer_hand_value": self.dealer.hand_value(),
            },
            indent=4
        )