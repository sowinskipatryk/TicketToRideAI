from typing import Tuple, List

from game.enums import ActionDecision, TrainCardDecision
from game.players.base_player import BasePlayer
from game.players.route_utils import (
    can_afford_route, best_color_for_grey, wilds_needed, find_best_affordable_route,
)
from game.ticket_deck import Ticket


class GreedyRouteAgent(BasePlayer):
    """Prioritizes claiming the highest-value routes it can afford.

    Strategy:
    - Always tries CLAIM_ROUTE first (picks highest point-value route)
    - Falls back to DRAW_CARDS if nothing is affordable
    - Draws cards matching colors it has most of (or wilds)
    - Keeps all tickets (greedy optimism)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._target_link_id = None
        self._target_color = None

    def decide_action(self) -> int:
        self._target_link_id = find_best_affordable_route(self, self.game)
        if self._target_link_id is not None:
            return ActionDecision.CLAIM_ROUTE.value
        return ActionDecision.DRAW_CARDS.value

    def decide_route(self) -> int:
        if self._target_link_id is not None:
            return self._target_link_id
        return 0

    def decide_cards_color(self) -> int:
        # Called for grey routes — pick the color we have the most of
        _, _, data = self.game.board.get_route_data(self._target_link_id)
        idx = best_color_for_grey(self.hand, self.game.config.TRAIN_COLORS, data['weight'])
        if idx is not None:
            self._target_color = self.game.config.TRAIN_COLORS[idx]
            return idx
        # Fallback: color with most cards
        best_idx = max(range(len(self.game.config.TRAIN_COLORS)),
                       key=lambda i: self.hand.get(self.game.config.TRAIN_COLORS[i], 0))
        self._target_color = self.game.config.TRAIN_COLORS[best_idx]
        return best_idx

    def decide_train_card(self) -> int:
        face_up = self.game.train_card_manager.get_face_up_cards()
        # Prefer wild cards
        for i, card in enumerate(face_up):
            if card == 'wild':
                return i
        # Prefer color we have the most of
        best_idx = TrainCardDecision.DRAW_PILE.value
        best_count = -1
        for i, card in enumerate(face_up):
            if card is not None and self.hand.get(card, 0) > best_count:
                best_count = self.hand.get(card, 0)
                best_idx = i
        return best_idx

    def decide_wild_cards(self) -> int:
        if self._target_link_id is None:
            return 0
        _, _, data = self.game.board.get_route_data(self._target_link_id)
        color = data['edge_color']
        if color == 'grey':
            color = self._target_color or self.game.config.TRAIN_COLORS[0]
        return wilds_needed(self.hand, color, data['weight'])

    def decide_tickets(self, min_keep: int, tickets: List[Ticket]) -> Tuple[List[int], List[int]]:
        all_ids = list(range(len(tickets)))
        return all_ids, []
