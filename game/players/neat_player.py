"""NEAT-based AI player using a neural network as a state evaluation function.

Uses 1-step greedy lookahead: for each legal action, simulate it on a cloned
SimState, encode the resulting state (same 389-feature encoding as AlphaZero),
evaluate with the NEAT network (single output in [-1, 1]), pick the best action.
"""
from typing import List, Tuple, TYPE_CHECKING

from neat.nn import FeedForwardNetwork

from game.enums import TrainCardDecision
from game.game_logger import logger
from game.players.base_player import BasePlayer
from game.players.route_utils import decide_tickets_by_distance
from game.ticket_deck import Ticket
from game.sim_game import from_game, get_legal_actions, clone, apply_action
from alphazero.encoding import encode_state

if TYPE_CHECKING:
    from game.core import Game


class NEATPlayer(BasePlayer):
    """NEAT-based AI player that uses a neural network as a state evaluation function.

    Action selection via 1-step greedy lookahead:
      1. Convert live Game to SimState via from_game()
      2. Get all legal actions
      3. For each action: clone state, apply action, encode, run NEAT network
      4. Pick the action with the highest value output (in [-1, 1])

    Uses the same 389-feature input encoding as AlphaZero so both methods
    see identical information about the game state.
    """

    def __init__(self, color_index: int, game: 'Game', network: FeedForwardNetwork = None):
        super().__init__(color_index, game)
        self.network = network
        self._chosen_action = None
        self._card_draw_count = 0

    def decide_action(self) -> int:
        """Run 1-step greedy lookahead, cache the best action, return its type."""
        state = from_game(self.game)
        legal = get_legal_actions(state)

        if not legal:
            self._chosen_action = None
            return 3  # SKIP

        best_action = None
        best_value = float('-inf')
        for action in legal:
            s = clone(state)
            apply_action(s, action)
            features = encode_state(s, self.player_id).tolist()
            value = self.network.activate(features)[0]
            if value > best_value:
                best_value = value
                best_action = action

        self._chosen_action = best_action
        self._card_draw_count = 0
        logger.debug(f'NEATPlayer chose action {best_action} (value={best_value:.3f})')
        return best_action.action_type

    def decide_route(self) -> int:
        if self._chosen_action and self._chosen_action.action_type == 0:
            return self._chosen_action.link_id
        return 0

    def decide_cards_color(self) -> int:
        if self._chosen_action and self._chosen_action.color:
            try:
                return self.game.config.TRAIN_COLORS.index(self._chosen_action.color)
            except ValueError:
                pass
        return max(range(len(self.game.config.TRAIN_COLORS)),
                   key=lambda i: self.hand.get(self.game.config.TRAIN_COLORS[i], 0))

    def decide_wild_cards(self) -> int:
        if self._chosen_action and self._chosen_action.action_type == 0:
            return self._chosen_action.wilds
        return 0

    def decide_train_card(self) -> int:
        if self._card_draw_count == 0 and self._chosen_action and self._chosen_action.action_type == 2:
            self._card_draw_count += 1
            choice = self._chosen_action.card_choice
            if 0 <= choice <= 5:
                return choice
        self._card_draw_count += 1
        return TrainCardDecision.DRAW_PILE.value

    def decide_tickets(self, min_keep: int, tickets: List[Ticket]) -> Tuple[List[int], List[int]]:
        return decide_tickets_by_distance(self, min_keep, tickets)
