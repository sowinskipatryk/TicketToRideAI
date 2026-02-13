import networkx as nx
from typing import Tuple, List, Dict, Optional

from game.enums import ActionDecision, TrainCardDecision
from game.players.base_player import BasePlayer
from game.players.route_utils import can_afford_route, best_color_for_grey, wilds_needed
from game.ticket_deck import Ticket


class TicketFocusedAgent(BasePlayer):
    """Prioritizes completing destination tickets using shortest-path planning.

    Strategy:
    - Plans shortest paths for incomplete tickets
    - Claims routes along planned paths first
    - Draws cards in colors needed for planned routes
    - Keeps tickets that share cities with existing ones (network synergy)
    - Draws new tickets when all current ones are completed
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._planned_routes: List[Dict] = []  # [{link_id, edge_color, weight}, ...]
        self._needed_colors: Dict[str, int] = {}
        self._target_link_id: Optional[int] = None
        self._target_color: Optional[str] = None

    def decide_action(self) -> int:
        self._update_plan()

        # If all tickets completed and deck has tickets, draw more
        if self.tickets:
            completed = sum(self.tickets.values())
            if completed == len(self.tickets) and self.game.ticket_deck.tickets_left > 0:
                return ActionDecision.DRAW_TICKETS.value

        # Try to claim a planned route
        self._target_link_id = self._find_claimable_planned_route()
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

        # Pick most-needed color from plan
        if self._needed_colors:
            best_color = max(self._needed_colors, key=self._needed_colors.get)
            if best_color in self.game.config.TRAIN_COLORS:
                self._target_color = best_color
                return self.game.config.TRAIN_COLORS.index(best_color)

        best_idx = max(range(len(self.game.config.TRAIN_COLORS)),
                       key=lambda i: self.hand.get(self.game.config.TRAIN_COLORS[i], 0))
        self._target_color = self.game.config.TRAIN_COLORS[best_idx]
        return best_idx

    def decide_train_card(self) -> int:
        face_up = self.game.train_card_manager.get_face_up_cards()

        # Prefer needed colors from face-up
        for i, card in enumerate(face_up):
            if card is not None and card in self._needed_colors:
                return i

        # Wild cards are always useful
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
        existing_cities = set()
        for ticket in self.tickets:
            existing_cities.add(ticket.city_from)
            existing_cities.add(ticket.city_to)

        scored = []
        for i, ticket in enumerate(tickets):
            score = 0.0
            if ticket.city_from in existing_cities:
                score += 2
            if ticket.city_to in existing_cities:
                score += 2
            score += ticket.points / 10.0

            try:
                shortest = nx.shortest_path_length(
                    self.game.board.G, ticket.city_from, ticket.city_to, weight='weight'
                )
                if shortest <= self.trains_remaining:
                    score += 1
                else:
                    score -= 3
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                score -= 5
            scored.append((i, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        keep_count = max(min_keep, sum(1 for _, s in scored if s > 0))
        keep_count = min(keep_count, len(tickets))

        indices = [i for i, _ in scored]
        return indices[:keep_count], indices[keep_count:]

    def _update_plan(self):
        """Recompute planned routes based on incomplete tickets."""
        self._planned_routes = []
        self._needed_colors = {}

        # Build simplified graph of available routes
        available = nx.Graph()
        link_data_map: Dict[tuple, List[Dict]] = {}

        for u, v, data in self.game.board.G.edges(data=True):
            owner = data['claimed_by']
            if owner is None or owner == self.color:
                w = data['weight']
                # Keep lightest edge for shortest path
                if not available.has_edge(u, v) or available[u][v]['weight'] > w:
                    available.add_edge(u, v, weight=w)
                link_data_map.setdefault((u, v), []).append(data)
                link_data_map.setdefault((v, u), []).append(data)

        for ticket, completed in self.tickets.items():
            if completed:
                continue
            try:
                path = nx.shortest_path(available, ticket.city_from, ticket.city_to, weight='weight')
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

            for i in range(len(path) - 1):
                candidates = link_data_map.get((path[i], path[i + 1]), [])
                for cand in candidates:
                    if cand['claimed_by'] is None:
                        self._planned_routes.append({
                            'link_id': cand['link_id'],
                            'edge_color': cand['edge_color'],
                            'weight': cand['weight'],
                        })
                        color = cand['edge_color']
                        if color != 'grey':
                            self._needed_colors[color] = self._needed_colors.get(color, 0) + cand['weight']
                        break

    def _find_claimable_planned_route(self) -> Optional[int]:
        """Find the first planned route we can afford and claim."""
        for route_info in self._planned_routes:
            link_id = route_info['link_id']
            _, _, data = self.game.board.get_route_data(link_id)
            if data['claimed_by'] is not None:
                continue
            if data['weight'] > self.trains_remaining:
                continue
            if not can_afford_route(self.hand, self.game.config.TRAIN_COLORS, data['edge_color'], data['weight']):
                continue
            if not self.game.board.validate_route(self.color, data):
                continue
            return link_id
        return None
