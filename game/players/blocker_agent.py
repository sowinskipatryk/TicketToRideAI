import networkx as nx
from typing import Tuple, List, Optional

from game.enums import ActionDecision, TrainCardDecision
from game.players.base_player import BasePlayer
from game.players.route_utils import (
    can_afford_route, best_color_for_grey, wilds_needed, find_best_affordable_route,
)
from game.ticket_deck import Ticket


class BlockerAgent(BasePlayer):
    """Monitors opponents and claims routes that block their likely paths.

    Strategy:
    - Analyzes opponents' claimed routes to find expansion endpoints (degree-1 nodes)
    - Claims unclaimed routes adjacent to those endpoints to block progress
    - Falls back to pursuing own tickets when no blocking opportunity exists
    - Draws cards (preferring wilds) when nothing is claimable
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._target_link_id: Optional[int] = None
        self._target_color: Optional[str] = None

    def decide_action(self) -> int:
        # Priority 1: block an opponent
        self._target_link_id = self._find_blocking_route()
        if self._target_link_id is not None:
            return ActionDecision.CLAIM_ROUTE.value

        # Priority 2: claim own best route
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
        for i, card in enumerate(face_up):
            if card == 'wild':
                return i
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
        # Keep short/easy tickets (best reward-to-distance ratio)
        scored = []
        for i, ticket in enumerate(tickets):
            try:
                dist = nx.shortest_path_length(
                    self.game.board.G, ticket.city_from, ticket.city_to, weight='weight'
                )
                scored.append((i, ticket.points / max(dist, 1)))
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                scored.append((i, -1))
        scored.sort(key=lambda x: x[1], reverse=True)
        indices = [i for i, _ in scored]
        min_keep = min(min_keep, len(tickets))
        return indices[:min_keep], indices[min_keep:]

    def _find_blocking_route(self) -> Optional[int]:
        """Find an affordable route that blocks an opponent's likely expansion."""
        for player in self.game.players:
            if player.player_id == self.player_id:
                continue

            subgraph = self.game.board.player_subgraph(player.color)
            if subgraph.number_of_edges() == 0:
                continue

            # Degree-1 nodes are likely expansion points
            endpoints = [n for n in subgraph.nodes() if subgraph.degree(n) == 1]

            for endpoint in endpoints:
                for u, v, data in self.game.board.G.edges(endpoint, data=True):
                    if data['claimed_by'] is not None:
                        continue
                    if data['weight'] > self.trains_remaining:
                        continue
                    if not can_afford_route(
                        self.hand, self.game.config.TRAIN_COLORS,
                        data['edge_color'], data['weight']
                    ):
                        continue
                    if not self.game.board.validate_route(self.color, data):
                        continue
                    return data['link_id']

        return None
