from typing import Optional, Dict, List, Tuple, TYPE_CHECKING

import networkx as nx

from game.ticket_deck import Ticket

if TYPE_CHECKING:
    from game.players.base_player import BasePlayer
    from game.core import Game


def can_afford_route(hand: Dict[str, int], train_colors: list, edge_color: str, weight: int) -> bool:
    """Check if a hand has enough cards (color + wild) for a route."""
    wild_count = hand.get('wild', 0)
    if edge_color == 'grey':
        return any(hand.get(c, 0) + wild_count >= weight for c in train_colors)
    return hand.get(edge_color, 0) + wild_count >= weight


def best_color_for_grey(hand: Dict[str, int], train_colors: list, weight: int) -> Optional[int]:
    """Pick the best color index to use for a grey route.

    Prefers the color where we have the most cards (minimizing wild usage).
    Returns None if no color works.
    """
    wild_count = hand.get('wild', 0)
    best_idx = None
    best_count = -1
    for idx, color in enumerate(train_colors):
        count = hand.get(color, 0)
        if count + wild_count >= weight and count > best_count:
            best_count = count
            best_idx = idx
    return best_idx


def wilds_needed(hand: Dict[str, int], color: str, weight: int) -> int:
    """Calculate how many wild cards are needed to claim a route of given color and weight."""
    color_cards = hand.get(color, 0)
    needed = max(0, weight - color_cards)
    return min(needed, hand.get('wild', 0))


def decide_tickets_by_distance(
    player: 'BasePlayer',
    min_keep: int,
    tickets: List[Ticket],
) -> Tuple[List[int], List[int]]:
    """Score tickets by reward-to-distance ratio and keep the best ones.

    Prefers tickets that are reachable within the player's remaining trains.
    Used by MCTS and AlphaZero players.

    Returns:
        Tuple of (kept_indices, discarded_indices).
    """
    scored = []
    for i, ticket in enumerate(tickets):
        try:
            dist = nx.shortest_path_length(
                player.game.board.G, ticket.city_from, ticket.city_to, weight='weight'
            )
            score = ticket.points / max(dist, 1)
            if dist <= player.trains_remaining:
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


def find_best_affordable_route(player: 'BasePlayer', game: 'Game', min_length: int = 1) -> Optional[int]:
    """Find the highest-value unclaimed route the player can claim.

    Returns link_id of best route, or None if no route is affordable.
    """
    best_link_id = None
    best_value = -1

    for u, v, data in game.board.G.edges(data=True):
        if data['claimed_by'] is not None:
            continue

        weight = data['weight']
        if weight < min_length or weight > player.trains_remaining:
            continue

        if not can_afford_route(player.hand, game.config.TRAIN_COLORS, data['edge_color'], weight):
            continue

        if not game.board.validate_route(player.color, data):
            continue

        value = game.get_route_value(weight)
        if value > best_value:
            best_value = value
            best_link_id = data['link_id']

    return best_link_id
