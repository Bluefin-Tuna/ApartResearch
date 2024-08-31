import random

CARDS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'jack', 'queen', 'king', 'ace']

class Blackjack:
    """
    Represents a game of Blackjack. 
    
    This class manages the game state, including players, dealer, and deck.

    Attributes:
        deck (Deck): The deck of cards used in the game.
        players (list): A list of Player objects representing the players.
        dealer (Dealer): The dealer for the game.
        current_player_index (int): Index of the current player.
        game_over (bool): Flag indicating if the game has ended.
        dealer_next_card (Card): The next card to be dealt to the dealer (for testing purposes).
    """

    def __init__(self, num_players):
        """
        Initializes a new Blackjack game.

        Args:
            num_players (int): The number of players in the game.
        """
        self.deck = Deck()
        self.players = [Player(f"Player {i+1}") for i in range(num_players)]
        self.dealer = Dealer()
        self.current_player_index = 0
        self.game_over = False
        self.dealer_next_card = None

    def deal_cards(self):
        """
        Deals initial cards to all players and the dealer.

        Each player and the dealer receive two cards.
        """
        for _ in range(2):
            for player in self.players:
                player.hit(self.deck.draw())
            self.dealer.hit(self.deck.draw())

    def get_game_state(self):
        """
        Retrieves the current state of the game.

        Returns:
            dict: A dictionary containing the current game state, including:
                - current_player: Name of the current player
                - current_player_hand: Current player's hand
                - current_player_hand_value: Value of current player's hand
                - dealer_visible_card: Dealer's visible card
                - other_players: List of other players' visible cards
                - game_over: Flag indicating if the game has ended
        """
        if self.current_player_index < len(self.players):
            current_player = self.players[self.current_player_index]
        else:
            current_player = self.dealer

        return {
            "name": current_player.name,
            "hand": current_player.show_hand(),
            "hand_value": current_player.get_hand_value(),
            "visible_card": str(self.dealer.hand[0]),
            "players": [
                {"name": p.name, "visible_cards": p.show_hand()} for p in self.players if p != current_player
            ],
            "game_over": self.game_over
        }

    def play_action(self, action):
        """
        Executes a player's or dealer's action.

        Args:
            action (str): The action to be performed ('hit' or 'stay').

        Returns:
            str: A message describing the result of the action.
        """
        if self.game_over:
            return "Game is over. Start a new game."

        if self.current_player_index < len(self.players):
            return self._player_action(action)
        else:
            return self._dealer_action(action)

    def _player_action(self, action):
        """
        Executes a player's action.

        Args:
            action (str): The action to be performed ('hit' or 'stay').

        Returns:
            str: A message describing the result of the action.
        """
        player = self.players[self.current_player_index]
        if action == 'hit':
            player.hit(self.deck.draw())
            if player.get_hand_value() > 21:
                self.current_player_index += 1
        elif action == 'stay':
            self.current_player_index += 1
        else:
            return "Invalid action. Please use 'hit' to hit or 'stay' to stay."

        if self.current_player_index == len(self.players):
            return "All players have finished. Dealer's turn."
        return f"{player.name}'s turn is over."

    def _dealer_action(self, action):
        """
        Executes the dealer's action.

        Args:
            action (str): The action to be performed ('hit' or 'stay').

        Returns:
            str: A message describing the result of the action.
        """
        if self.dealer.get_hand_value() >= 17:
            self.game_over = True
            return "Dealer's turn is over. Game finished."

        if action == 'hit':
            if self.dealer_next_card:
                card = self.dealer_next_card
                self.dealer_next_card = None
            else:
                card = self.deck.draw()
            self.dealer.hit(card)
            if self.dealer.get_hand_value() > 21:
                self.game_over = True
                return "Dealer busts. Game over."
        elif action == 'stay':
            self.game_over = True
            return "Dealer stays. Game over."
        else:
            return "Invalid action. Please use 'hit' to hit or 'stay' to stay."

        return "Dealer's turn continues."

    def show_results(self):
        """
        Displays the final results of the game.

        Returns:
            str: A string containing the game results, including each player's hand, 
             hand value, and whether they won, lost, or tied with the dealer.
        """
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
    
    def set_dealer_next_card(self, card):
        """
        Sets the next card to be dealt to the dealer.

        Args:
            card (Card): The card to be dealt to the dealer next.
        """
        self.dealer_next_card = card
        
class Card:
    """
    Represents a playing card.

    Attributes:
        name (str): The name of the card (e.g., "Ace", "King", "7").
        value (int): The numerical value of the card in Blackjack.
    """

    def __init__(self, name, value):
        """
        Initializes a new Card.

        Args:
            name (str): The name of the card.
            value (int): The numerical value of the card.
        """
        self.name = name
        self.value = value

    def __str__(self):
        """
        Returns a string representation of the card.

        Returns:
            str: A string representation of the card, including its name and value.
        """
        faces = ['jack', 'queen', 'king', 'ace']
        return  f"{self.name} ({self.value})" if self.name in faces else self.name

class Deck:
    """
    Represents a deck of playing cards.

    Attributes:
        cards (list): A list of tuples representing the cards in the deck.
    """

    def __init__(self):
        """
        Initializes a new Deck with a standard set of 52 cards.
        """
        self.cards = [
            ("2", 2), ("3", 3), ("4", 4), ("5", 5), ("6", 6), ("7", 7), ("8", 8), ("9", 9), ("10", 10),
            ("jack", 10), ("queen", 10), ("king", 10), ("ace", 11)
        ]

    def draw(self):
        """
        Draws a random card from the deck.

        Returns:
            Card: A randomly selected Card object.
        """
        name, value = random.choice(self.cards)
        return Card(name, value)

    def get_value(self, name):
        """
        Returns the value of a card given the name.
        """
        return next((value for card, value in self.cards if card.lower() == name.lower()), None)

class Player:
    """
    Represents a player in the Blackjack game.

    Attributes:
        name (str): The name of the player.
        hand (list): A list of Card objects representing the player's hand.
    """

    def __init__(self, name):
        """
        Initializes a new Player.

        Args:
            name (str): The name of the player.
        """
        self.name = name
        self.hand = []

    def hit(self, card):
        """
        Adds a card to the player's hand.

        Args:
            card (Card): The card to be added to the player's hand.
        """
        self.hand.append(card)

    def get_hand_value(self):
        """
        Calculates the total value of the player's hand.

        Returns:
            int: The total value of the player's hand, accounting for Aces.
        """
        value = sum(card.value for card in self.hand)
        aces = sum(1 for card in self.hand if card.name == "ace")
        while value > 21 and aces:
            value -= 10
            aces -= 1
        return value

    def show_hand(self):
        """
        Returns a string representation of the player's hand.

        Returns:
            str: A string showing all cards in the player's hand.
        """
        return ', '.join(str(card) for card in self.hand)

class Dealer(Player):
    """
    Represents the dealer in the Blackjack game.

    Inherits from Player class and adds dealer specific behavior.
    """

    def __init__(self):
        """
        Initializes a new Dealer.
        """
        super().__init__("Dealer")

    def show_hand(self, hide_card=True):
        """
        Returns a string representation of the dealer's hand.

        Args:
            hide_card (bool): If True, hides the second card of the dealer's hand.

        Returns:
            str: A string showing the dealer's visible cards.
        """
        if hide_card and len(self.hand) > 1:
            return f"{self.hand[0]}, <hidden>"
        return super().show_hand()