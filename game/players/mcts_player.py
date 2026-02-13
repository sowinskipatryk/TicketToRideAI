import networkx as nx
from typing import Tuple, List

from game.enums import TrainCardDecision
from game.players.base_player import BasePlayer
from game.ticket_deck import Ticket
from mcts.is_mcts import ISMCTS
from mcts.rollout_policies import heuristic_rollout


class MCTSPlayer(BasePlayer):
    """Player that uses Information Set MCTS for decision-making.

    Runs IS-MCTS search on decide_action() and caches the chosen Action.
    Subsequent decision methods (decide_route, decide_cards_color, etc.)
    read from the cached action.
    """

    DEFAULT_ITERATIONS = 400

    def __init__(self, color_index, game, adapter, iterations=None):
        super().__init__(color_index, game, adapter)
        iters = iterations or self.DEFAULT_ITERATIONS
        self.mcts = ISMCTS(
            iterations=iters,
            rollout_policy=heuristic_rollout,
            max_rollout_depth=60,
        )
        self._chosen_action = None
        self._card_draw_count = 0

    def decide_action(self) -> int:
        self._chosen_action = self.mcts.search(self.game, self.player_id)
        self._card_draw_count = 0
        return self._chosen_action.action_type

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
        scored = []
        for i, ticket in enumerate(tickets):
            try:
                dist = nx.shortest_path_length(
                    self.game.board.G, ticket.city_from, ticket.city_to, weight='weight'
                )
                score = ticket.points / max(dist, 1)
                if dist <= self.trains_remaining:
                    score += 1
                else:
                    score -= 2
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                score = -1
            scored.append((i, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        keep_count = max(min_keep, sum(1 for _, s in scored if s > 0))
        keep_count = min(keep_count, len(tickets))

        indices = [i for i, _ in scored]
        return indices[:keep_count], indices[keep_count:]
