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

    Strategy:
    1. Always claim a route that completes a ticket
    2. Only claim routes worth 4+ points, otherwise draw cards
    3. Prefer drawing needed colors from face-up, then wilds, then pile
    4. Draw tickets if hand is large and all tickets completed
    """
    pid = state.current_player
    hand = state.hands[pid]
    hand_size = sum(hand.values())
    player_tickets = state.tickets[pid]

    # Separate action types
    claim_actions = [a for a in actions if a.action_type == 0]
    draw_actions = [a for a in actions if a.action_type == 2]
    ticket_actions = [a for a in actions if a.action_type == 1]

    # 1. Always claim a route that completes a ticket
    if claim_actions:
        for action in claim_actions:
            for ticket, completed in player_tickets.items():
                if completed:
                    continue
                if _would_complete_ticket(state, pid, action.link_id, ticket):
                    return action

    # 2. Claim high-value routes (4+ weight = 7+ points)
    good_claims = []
    for a in claim_actions:
        ri = _find_route(state, a.link_id)
        if ri and ri.weight >= 4:
            good_claims.append((a, state.route_values.get(ri.weight, 0)))
    if good_claims:
        good_claims.sort(key=lambda x: x[1], reverse=True)
        return good_claims[0][0]

    # 3. If we have enough cards, claim medium routes (3+)
    if hand_size >= 8:
        medium_claims = []
        for a in claim_actions:
            ri = _find_route(state, a.link_id)
            if ri and ri.weight >= 3:
                medium_claims.append((a, state.route_values.get(ri.weight, 0)))
        if medium_claims:
            medium_claims.sort(key=lambda x: x[1], reverse=True)
            return medium_claims[0][0]

    # 4. Draw tickets if all current tickets completed and we have cards
    incomplete = sum(1 for _, c in player_tickets.items() if not c)
    if ticket_actions and incomplete == 0 and hand_size >= 6:
        return ticket_actions[0]

    # 5. Draw cards — prefer needed colors, then wilds, then pile
    if draw_actions:
        needed_colors = _get_needed_colors(state, pid)

        # Face-up wilds
        for action in draw_actions:
            if action.card_choice < 5:
                card = state.face_up_cards[action.card_choice]
                if card == 'wild':
                    return action

        # Needed colors from face-up
        for action in draw_actions:
            if action.card_choice < 5:
                card = state.face_up_cards[action.card_choice]
                if card in needed_colors:
                    return action

        # Draw pile
        pile_action = next((a for a in draw_actions if a.card_choice == 5), None)
        if pile_action:
            return pile_action

    # 6. Claim any remaining route (even small ones) as last resort
    if claim_actions:
        best = max(claim_actions,
                   key=lambda a: state.route_values.get(
                       _find_route(state, a.link_id).weight, 0)
                   if _find_route(state, a.link_id) else 0)
        return best

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
