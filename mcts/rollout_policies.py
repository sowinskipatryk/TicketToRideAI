"""Rollout policies for MCTS simulation phase.

These functions select actions during the simulation (playout) phase of MCTS.
They operate on SimState directly — no dependency on BasePlayer or Game.
"""
import random
from collections import defaultdict, deque
from typing import List, Dict, Set

from game.ticket_deck import Ticket
from mcts.sim_game import SimState, Action, _is_ticket_completed


def random_rollout(state: SimState, actions: List[Action]) -> Action:
    """Pure random action selection."""
    return random.choice(actions)


def heuristic_rollout(state: SimState, actions: List[Action]) -> Action:
    """Ticket-aware heuristic rollout.

    Priority:
    1. Claim a route that completes a ticket
    2. Claim a route on the path toward an incomplete ticket
    3. Claim the highest-value route available
    4. Draw a needed card color from face-up
    5. Draw from pile
    6. Random
    """
    pid = state.current_player
    hand = state.hands[pid]
    player_tickets = state.tickets[pid]

    # Separate action types
    claim_actions = [a for a in actions if a.action_type == 0]
    draw_actions = [a for a in actions if a.action_type == 2]

    if claim_actions:
        # Check if any claim completes a ticket
        for action in claim_actions:
            ri = _find_route(state, action.link_id)
            if ri is None:
                continue
            # Simulate this claim and check ticket completion
            for ticket, completed in player_tickets.items():
                if completed:
                    continue
                if _would_complete_ticket(state, pid, action.link_id, ticket):
                    return action

        # Pick highest-value claimable route
        best_action = max(claim_actions,
                          key=lambda a: state.route_values.get(
                              _find_route(state, a.link_id).weight, 0)
                          if _find_route(state, a.link_id) else 0)
        if best_action:
            return best_action

    if draw_actions:
        # Prefer drawing cards in colors we need
        needed_colors = _get_needed_colors(state, pid)

        # Check face-up cards for needed colors
        for action in draw_actions:
            if action.card_choice < 5:
                card = state.face_up_cards[action.card_choice]
                if card == 'wild':
                    return action  # Always grab wilds
                if card in needed_colors:
                    return action

        # Default to draw pile
        pile_action = next((a for a in draw_actions if a.card_choice == 5), None)
        if pile_action:
            return pile_action

    return random.choice(actions)


def _find_route(state: SimState, link_id: int):
    """Find RouteInfo by link_id."""
    for ri in state.route_info:
        if ri.link_id == link_id:
            return ri
    return None


def _would_complete_ticket(state: SimState, pid: int, new_link_id: int, ticket: Ticket) -> bool:
    """Check if claiming new_link_id would complete the ticket for player pid."""
    # Build adjacency including the new link
    adj: Dict[str, List[str]] = defaultdict(list)
    for ri in state.route_info:
        if state.route_owners.get(ri.link_id) == pid or ri.link_id == new_link_id:
            adj[ri.city_from].append(ri.city_to)
            adj[ri.city_to].append(ri.city_from)

    if ticket.city_from not in adj:
        return False

    # BFS
    visited = {ticket.city_from}
    queue = deque([ticket.city_from])
    while queue:
        city = queue.popleft()
        if city == ticket.city_to:
            return True
        for neighbor in adj[city]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return False


def _get_needed_colors(state: SimState, pid: int) -> Set[str]:
    """Determine which card colors the player needs for incomplete tickets.

    Looks at the shortest unclaimed route segments toward incomplete tickets.
    """
    needed: Set[str] = set()

    for ticket, completed in state.tickets[pid].items():
        if completed:
            continue

        # Find unclaimed routes on path toward ticket
        # Simple approach: look at routes adjacent to cities we've claimed routes to
        my_cities: Set[str] = set()
        for ri in state.route_info:
            if state.route_owners.get(ri.link_id) == pid:
                my_cities.add(ri.city_from)
                my_cities.add(ri.city_to)

        # Add ticket endpoints
        target_cities = {ticket.city_from, ticket.city_to}
        relevant_cities = my_cities | target_cities

        for ri in state.route_info:
            if state.route_owners.get(ri.link_id) is not None:
                continue
            if ri.city_from in relevant_cities or ri.city_to in relevant_cities:
                if ri.edge_color != 'grey':
                    needed.add(ri.edge_color)

    return needed
