import random
from typing import List, Dict, Any, Tuple
from pyfiles.framework import GameEnvironment, ControlledMechanic, AgentController

class Card:
    def __init__(self, name: str, value: int):
        self.name = name
        self.value = value

    def __str__(self):
        return f"{self.name} ({self.value})"

class Deck:
    def __init__(self):
        self.cards = [
            ("2", 2), ("3", 3), ("4", 4), ("5", 5), ("6", 6), ("7", 7), ("8", 8), ("9", 9), ("10", 10),
            ("jack", 10), ("queen", 10), ("king", 10), ("ace", 11)
        ]

    def draw(self) -> Card:
        name, value = random.choice(self.cards)
        return Card(name, value)

class Player:
    def __init__(self, name: str):
        self.name = name
        self.hand: List[Card] = []

    def hit(self, card: Card):
        self.hand.append(card)

    def get_hand_value(self) -> int:
        value = sum(card.value for card in self.hand)
        aces = sum(1 for card in self.hand if card.name == "ace")
        while value > 21 and aces:
            value -= 10
            aces -= 1
        return value

    def show_hand(self) -> str:
        return ', '.join(str(card) for card in self.hand)

class Dealer(Player):
    def __init__(self):
        super().__init__("Dealer")

    def show_hand(self, hide_card: bool = True) -> str:
        if hide_card and len(self.hand) > 1:
            return f"{self.hand[0]}, <hidden>"
        return super().show_hand()

class DealerDrawMechanic(ControlledMechanic):
    def __init__(self, deck: Deck):
        self.deck = deck

    def get_options(self, state: Dict[str, Any]) -> List[str]:
        return ["hit", "stay"]

    def execute_action(self, state: Dict[str, Any], action: str) -> Dict[str, Any]:
        if action == "hit":
            card = self.deck.draw()
            state["dealer"].hit(card)
        return state

class BlackjackEnvglobal(GameEnvironment):
    def __init__(self, num_players: int):
        super().__init__()
        self.deck = Deck()
        self.players = [Player(f"Player {i+1}") for i in range(num_players)]
        self.dealer = Dealer()
        self.current_player_index = 0
        self.game_over = False
        self.controlled_mechanics["dealer_draw"] = DealerDrawMechanic(self.deck)

    def reset(self) -> Dict[str, Any]:
        self.deck = Deck()
        for player in self.players:
            player.hand = []
        self.dealer.hand = []
        self.current_player_index = 0
        self.game_over = False
        self.deal_cards()
        return self.get_state()

    def deal_cards(self):
        for _ in range(2):
            for player in self.players:
                player.hit(self.deck.draw())
            self.dealer.hit(self.deck.draw())

    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self.game_over:
            return self.get_state(), 0, True, {"message": "Game is already over."}

        if self.current_player_index < len(self.players):
            return self._player_action(action)
        else:
            return self._dealer_action(action)

    def _player_action(self, action: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        player = self.players[self.current_player_index]
        if action == 'hit':
            player.hit(self.deck.draw())
            if player.get_hand_value() > 21:
                self.current_player_index += 1
                return self.get_state(), -1, False, {"message": f"{player.name} busts!"}
        elif action == 'stay':
            self.current_player_index += 1
        else:
            return self.get_state(), 0, False, {"message": "Invalid action."}

        if self.current_player_index == len(self.players):
            return self.get_state(), 0, False, {"message": "Dealer's turn."}
        return self.get_state(), 0, False, {"message": f"{player.name}'s turn is over."}

    def _dealer_action(self, action: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self.dealer.get_hand_value() >= 17:
            self.game_over = True
            return self.get_state(), 0, True, {"message": "Game finished."}

        if action == 'hit':
            self.dealer.hit(self.deck.draw())
            if self.dealer.get_hand_value() > 21:
                self.game_over = True
                return self.get_state(), 1, True, {"message": "Dealer busts. Game over."}
        elif action == 'stay':
            self.game_over = True
            return self.get_state(), 0, True, {"message": "Dealer stays. Game over."}
        else:
            return self.get_state(), 0, False, {"message": "Invalid action."}

        return self.get_state(), 0, False, {"message": "Dealer's turn continues."}

    def get_state(self) -> Dict[str, Any]:
        return {
            "players": [{"name": p.name, "hand": p.show_hand(), "value": p.get_hand_value()} for p in self.players],
            "dealer": {"hand": self.dealer.show_hand(hide_card=not self.game_over), "value": self.dealer.get_hand_value()},
            "current_player": self.current_player_index if self.current_player_index < len(self.players) else "Dealer",
            "game_over": self.game_over
        }

    def is_game_over(self) -> bool:
        return self.game_over

    def get_result(self) -> Dict[str, Any]:
        if not self.game_over:
            return {"message": "Game is not over yet."}

        dealer_value = self.dealer.get_hand_value()
        results = {
            "dealer": {"hand": self.dealer.show_hand(), "value": dealer_value},
            "players": []
        }

        for player in self.players:
            player_value = player.get_hand_value()
            player_result = {
                "name": player.name,
                "hand": player.show_hand(),
                "value": player_value
            }

            if player_value > 21:
                player_result["outcome"] = "Bust"
            elif dealer_value > 21 or player_value > dealer_value:
                player_result["outcome"] = "Win"
            elif player_value == dealer_value:
                player_result["outcome"] = "Tie"
            else:
                player_result["outcome"] = "Lose"

            results["players"].append(player_result)

        return results