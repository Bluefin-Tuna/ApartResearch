import random

class Blackjack:
    def __init__(self, num_players):
        self.deck = Deck()
        self.players = [Player(f"Player {i+1}") for i in range(num_players)]
        self.dealer = Dealer()
        self.current_player_index = 0
        self.game_over = False

    def deal_cards(self):
        for _ in range(2):
            for player in self.players:
                player.hit(self.deck.draw())
            self.dealer.hit(self.deck.draw())

    def get_game_state(self):
        if self.current_player_index < len(self.players):
            current_player = self.players[self.current_player_index]
        else:
            current_player = self.dealer

        return {
            "current_player": current_player.name,
            "current_player_hand": current_player.show_hand(),
            "current_player_hand_value": current_player.get_hand_value(),
            "dealer_visible_card": str(self.dealer.hand[0]),
            "other_players": [
                {"name": p.name, "visible_cards": p.show_hand()} for p in self.players if p != current_player
            ],
            "game_over": self.game_over
        }

    def play_action(self, action):
        if self.game_over:
            return "Game is over. Start a new game."

        if self.current_player_index < len(self.players):
            return self._player_action(action)
        else:
            return self._dealer_action(action)

    def _player_action(self, action):
        player = self.players[self.current_player_index]
        if action == 'hit':
            player.hit(self.deck.draw())
            if player.get_hand_value() > 21:
                print(f"{player.name} busts!")
                self.current_player_index += 1
        elif action == 'stay':
            self.current_player_index += 1
        else:
            return "Invalid action. Please use 'hit' to hit or 'stay' to stay."

        if self.current_player_index == len(self.players):
            return "All players have finished. Dealer's turn."
        return f"{player.name}'s turn is over."

    def _dealer_action(self, action):
        if self.dealer.get_hand_value() >= 17:
            self.game_over = True
            return "Dealer's turn is over. Game finished."

        if action == 'hit':
            self.dealer.hit(self.deck.draw())
            if self.dealer.get_hand_value() > 21:
                print("Dealer busts!")
                self.game_over = True
                return "Dealer busts. Game over."
        elif action == 'stay':
            self.game_over = True
            return "Dealer stays. Game over."
        else:
            return "Invalid action. Please use 'hit' to hit or 'stay' to stay."

        return "Dealer's turn continues."

    def show_results(self):
        if not self.game_over:
            return "Game is not over yet."

        dealer_value = self.dealer.get_hand_value()
        results = [f"Dealer's hand: {self.dealer.show_hand()} (Value: {dealer_value})"]
        
        for player in self.players:
            player_value = player.get_hand_value()
            results.append(f"{player.name}'s hand: {player.show_hand()} (Value: {player_value})")
            if player_value > 21:
                results.append(f"{player.name} busts!")
            elif dealer_value > 21 or player_value > dealer_value:
                results.append(f"{player.name} wins!")
            elif player_value == dealer_value:
                results.append(f"{player.name} ties with the dealer.")
            else:
                results.append(f"{player.name} loses.")
        
        return "\n".join(results)
    
class Card:
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __str__(self):
        faces = ['Jack', 'Queen', 'King', 'Ace']
        return  f"{self.name} ({self.value})" if self.name in faces else self.name

class Deck:
    def __init__(self):
        self.cards = [
            ("2", 2), ("3", 3), ("4", 4), ("5", 5), ("6", 6), ("7", 7), ("8", 8), ("9", 9), ("10", 10),
            ("Jack", 10), ("Queen", 10), ("King", 10), ("Ace", 11)
        ]

    def draw(self):
        name, value = random.choice(self.cards)
        return Card(name, value)

class Player:
    def __init__(self, name):
        self.name = name
        self.hand = []

    def hit(self, card):
        self.hand.append(card)
        print(f"{self.name} draws: {card}")

    def get_hand_value(self):
        value = sum(card.value for card in self.hand)
        aces = sum(1 for card in self.hand if card.name == "Ace")
        while value > 21 and aces:
            value -= 10
            aces -= 1
        return value

    def show_hand(self):
        return ', '.join(str(card) for card in self.hand)

class Dealer(Player):
    def __init__(self):
        super().__init__("Dealer")

    def show_hand(self, hide_card=True):
        if hide_card and len(self.hand) > 1:
            return f"{self.hand[0]}, <hidden>"
        return super().show_hand()