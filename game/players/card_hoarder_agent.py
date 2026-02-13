from typing import Tuple, List, Optional

from game.enums import ActionDecision, TrainCardDecision
from game.players.base_player import BasePlayer
from game.players.route_utils import (
    find_best_affordable_route, best_color_for_grey, wilds_needed,
)
from game.ticket_deck import Ticket


class CardHoarderAgent(BasePlayer):
    """Draws cards aggressively early, then claims routes in bulk.

    Strategy:
    - Phase 1 (hoarding): Draw cards until hand >= 15 or trains <= 30
      - Exception: claim 5+ length routes if affordable (too good to pass up)
    - Phase 2 (spending): Claim highest-value routes, draw only if nothing affordable
    - Keeps minimum tickets, prefers high-point ones
    - Always grabs wild cards from face-up
    """

    HOARD_THRESHOLD_TRAINS = 30
    HOARD_THRESHOLD_HAND = 15

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._target_link_id: Optional[int] = None
        self._target_color: Optional[str] = None

    def _is_hoarding_phase(self) -> bool:
        total_hand = sum(self.hand.values())
        return (self.trains_remaining > self.HOARD_THRESHOLD_TRAINS
                and total_hand < self.HOARD_THRESHOLD_HAND)

    def decide_action(self) -> int:
        if self._is_hoarding_phase():
            # Only claim if we find a 5+ length route (too valuable to skip)
            self._target_link_id = find_best_affordable_route(self, self.game, min_length=5)
            if self._target_link_id is not None:
                return ActionDecision.CLAIM_ROUTE.value
            return ActionDecision.DRAW_CARDS.value
        else:
            self._target_link_id = find_best_affordable_route(self, self.game)
            if self._target_link_id is not None:
                return ActionDecision.CLAIM_ROUTE.value
            return ActionDecision.DRAW_CARDS.value

    def decide_route(self) -> int:
        if self._target_link_id is not None:
            return self._target_link_id
        return 0

    def decide_cards_color(self) -> int:
        if self._target_link_id is not None:
            _, _, data = self.game.board.get_route_data(self._target_link_id)
            idx = best_color_for_grey(self.hand, self.game.config.TRAIN_COLORS, data['weight'])
            if idx is not None:
                self._target_color = self.game.config.TRAIN_COLORS[idx]
                return idx
        best_idx = max(range(len(self.game.config.TRAIN_COLORS)),
                       key=lambda i: self.hand.get(self.game.config.TRAIN_COLORS[i], 0))
        self._target_color = self.game.config.TRAIN_COLORS[best_idx]
        return best_idx

    def decide_train_card(self) -> int:
        face_up = self.game.train_card_manager.get_face_up_cards()
        # Always grab wilds
        for i, card in enumerate(face_up):
            if card == 'wild':
                return i
        # Otherwise draw from pile (more variety during hoarding)
        return TrainCardDecision.DRAW_PILE.value

    def decide_wild_cards(self) -> int:
        if self._target_link_id is None:
            return 0
        _, _, data = self.game.board.get_route_data(self._target_link_id)
        color = data['edge_color']
        if color == 'grey':
            color = self._target_color or self.game.config.TRAIN_COLORS[0]
        return wilds_needed(self.hand, color, data['weight'])

    def decide_tickets(self, min_keep: int, tickets: List[Ticket]) -> Tuple[List[int], List[int]]:
        # Keep minimum, prefer highest-point tickets
        scored = sorted(range(len(tickets)), key=lambda i: tickets[i].points, reverse=True)
        min_keep = min(min_keep, len(tickets))
        return scored[:min_keep], scored[min_keep:]
