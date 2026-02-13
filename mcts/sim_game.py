"""Lightweight game simulation engine for MCTS.

Operates on plain data (dicts, lists) for fast cloning and simulation.
No dependency on Game, BasePlayer, NetworkX, adapters, or logging.
"""
import copy
import random
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, TYPE_CHECKING

from game.ticket_deck import Ticket
from game.config import ConfigFactory

if TYPE_CHECKING:
    from game.core import Game


@dataclass
class Action:
    """A single atomic action in the simulation."""
    action_type: int  # 0=CLAIM_ROUTE, 1=DRAW_TICKETS, 2=DRAW_CARDS
    link_id: int = -1
    color: str = ''        # Color of cards to use (resolved for grey routes)
    wilds: int = 0
    card_choice: int = -1  # For DRAW_CARDS: face-up position (0-4) or 5=draw_pile

    def __eq__(self, other):
        if not isinstance(other, Action):
            return False
        return (self.action_type == other.action_type
                and self.link_id == other.link_id
                and self.color == other.color
                and self.wilds == other.wilds
                and self.card_choice == other.card_choice)

    def __hash__(self):
        return hash((self.action_type, self.link_id, self.color, self.wilds, self.card_choice))

    def __repr__(self):
        if self.action_type == 0:
            return f"ClaimRoute(link={self.link_id}, color={self.color}, wilds={self.wilds})"
        elif self.action_type == 1:
            return f"DrawTickets()"
        elif self.action_type == 2:
            return f"DrawCards(choice={self.card_choice})"
        return f"Action({self.action_type})"


# Route data entry: static, shared across all SimState copies
@dataclass(frozen=True)
class RouteInfo:
    link_id: int
    route_id: int
    city_from: str
    city_to: str
    weight: int
    edge_color: str


class SimState:
    """Complete game state for simulation. All fields are plain data.

    Designed for fast deepcopy — route_info is shared (frozen), everything
    else is plain dicts/lists.
    """
    __slots__ = [
        'route_owners', 'route_info', 'route_ids_by_route_id',
        'draw_pile', 'discard_pile', 'face_up_cards',
        'ticket_deck_remaining',
        'hands', 'tickets', 'scores', 'trains',
        'current_player', 'game_phase', 'last_player', 'num_players',
        'config', 'route_values', 'wild_card_restriction', 'min_train_figures',
        'cards_per_draw',
    ]

    def __init__(self):
        self.route_owners: Dict[int, Optional[int]] = {}
        self.route_info: List[RouteInfo] = []  # Static, shared
        self.route_ids_by_route_id: Dict[int, List[int]] = {}  # route_id → [link_ids]

        self.draw_pile: List[str] = []
        self.discard_pile: List[str] = []
        self.face_up_cards: List[Optional[str]] = [None] * 5

        self.ticket_deck_remaining: List[Ticket] = []

        self.hands: List[Dict[str, int]] = []
        self.tickets: List[Dict[Ticket, bool]] = []
        self.scores: List[int] = []
        self.trains: List[int] = []

        self.current_player: int = 0
        self.game_phase: str = 'running'  # 'running', 'last_round', 'finished'
        self.last_player: Optional[int] = None
        self.num_players: int = 2

        self.config: Any = None  # Reference to config (not cloned)
        self.route_values: Dict[int, int] = {}
        self.wild_card_restriction: bool = True
        self.min_train_figures: int = 3
        self.cards_per_draw: int = 2


def clone(state: SimState) -> SimState:
    """Deep copy a SimState, sharing static route_info."""
    s = SimState()
    s.route_owners = state.route_owners.copy()
    s.route_info = state.route_info  # Shared, frozen
    s.route_ids_by_route_id = state.route_ids_by_route_id  # Shared, static

    s.draw_pile = state.draw_pile.copy()
    s.discard_pile = state.discard_pile.copy()
    s.face_up_cards = state.face_up_cards.copy()

    s.ticket_deck_remaining = state.ticket_deck_remaining.copy()

    s.hands = [dict(h) for h in state.hands]
    s.tickets = [dict(t) for t in state.tickets]
    s.scores = state.scores.copy()
    s.trains = state.trains.copy()

    s.current_player = state.current_player
    s.game_phase = state.game_phase
    s.last_player = state.last_player
    s.num_players = state.num_players

    s.config = state.config
    s.route_values = state.route_values
    s.wild_card_restriction = state.wild_card_restriction
    s.min_train_figures = state.min_train_figures
    s.cards_per_draw = state.cards_per_draw
    return s


def from_game(game: 'Game') -> SimState:
    """Extract a SimState from a live Game object."""
    s = SimState()
    s.num_players = game.players_num
    s.config = game.config
    s.route_values = dict(game.config.ROUTE_VALUES)
    s.wild_card_restriction = getattr(game.config, 'WILD_CARD_RESTRICTION', True)
    s.min_train_figures = game.config.MIN_TRAIN_FIGURES
    s.cards_per_draw = game.config.NUM_TRAIN_CARDS_DEALT

    # Board
    route_info_list = []
    route_ids_map: Dict[int, List[int]] = {}
    for u, v, data in game.board.G.edges(data=True):
        ri = RouteInfo(
            link_id=data['link_id'],
            route_id=data['route_id'],
            city_from=u,
            city_to=v,
            weight=data['weight'],
            edge_color=data['edge_color'],
        )
        route_info_list.append(ri)
        s.route_owners[data['link_id']] = (
            _player_color_to_id(data['claimed_by'], game) if data['claimed_by'] else None
        )
        route_ids_map.setdefault(data['route_id'], []).append(data['link_id'])

    s.route_info = route_info_list
    s.route_ids_by_route_id = route_ids_map

    # Cards
    s.draw_pile = list(game.train_card_manager._draw_pile)
    s.discard_pile = list(game.train_card_manager._discard_pile)
    s.face_up_cards = list(game.train_card_manager._face_up_cards)

    # Tickets
    s.ticket_deck_remaining = list(game.ticket_deck._ticket_deck)

    # Per-player
    for player in game.players:
        s.hands.append(dict(player.hand))
        s.tickets.append(dict(player.tickets))
        s.scores.append(player.score)
        s.trains.append(player.trains_remaining)

    # Game flow
    s.current_player = game.current_player_id
    if game.game_state.name == 'LAST_ROUND':
        s.game_phase = 'last_round'
        s.last_player = game.last_player.player_id if game.last_player else None
    elif game.game_state.name == 'FINISHED':
        s.game_phase = 'finished'
    else:
        s.game_phase = 'running'
    if game.last_player:
        s.last_player = game.last_player.player_id

    return s


def _player_color_to_id(color, game: 'Game') -> Optional[int]:
    """Convert PlayerColor enum to player_id."""
    for p in game.players:
        if p.color == color:
            return p.player_id
    return None


def is_terminal(state: SimState) -> bool:
    return state.game_phase == 'finished'


def get_legal_actions(state: SimState) -> List[Action]:
    """Enumerate all valid actions for the current player."""
    if is_terminal(state):
        return []

    pid = state.current_player
    hand = state.hands[pid]
    trains = state.trains[pid]
    actions = []

    # CLAIM_ROUTE actions
    for ri in state.route_info:
        if state.route_owners[ri.link_id] is not None:
            continue
        if ri.weight > trains:
            continue
        if not _validate_route(state, pid, ri):
            continue

        if ri.edge_color == 'grey':
            # For grey routes, pick only the best color (fewest wilds needed)
            best_color = None
            best_wilds = None
            for color in state.config.TRAIN_COLORS:
                wilds = _wilds_needed(hand, color, ri.weight)
                if wilds is not None and (best_wilds is None or wilds < best_wilds):
                    best_color = color
                    best_wilds = wilds
            if best_color is not None:
                actions.append(Action(
                    action_type=0, link_id=ri.link_id,
                    color=best_color, wilds=best_wilds,
                ))
        else:
            wilds = _wilds_needed(hand, ri.edge_color, ri.weight)
            if wilds is not None:
                actions.append(Action(
                    action_type=0, link_id=ri.link_id,
                    color=ri.edge_color, wilds=wilds,
                ))

    # DRAW_CARDS actions — deduplicate by card color seen in face-up
    has_cards = len(state.draw_pile) > 0 or len(state.discard_pile) > 0
    seen_colors = set()
    wild_face_up_idx = None
    for i, card in enumerate(state.face_up_cards):
        if card is None:
            continue
        if card == 'wild':
            if wild_face_up_idx is None:
                wild_face_up_idx = i
            continue
        if card not in seen_colors:
            seen_colors.add(card)
            actions.append(Action(action_type=2, card_choice=i))
    # Wild face-up as separate option (costs the whole turn)
    if wild_face_up_idx is not None:
        actions.append(Action(action_type=2, card_choice=wild_face_up_idx))
    if has_cards:
        actions.append(Action(action_type=2, card_choice=5))  # draw pile

    # DRAW_TICKETS action
    if len(state.ticket_deck_remaining) > 0:
        actions.append(Action(action_type=1))

    return actions


def apply_action(state: SimState, action: Action) -> None:
    """Apply an action to the state in-place. Advances to next player."""
    pid = state.current_player

    if action.action_type == 0:
        _apply_claim_route(state, pid, action)
    elif action.action_type == 1:
        _apply_draw_tickets(state, pid)
    elif action.action_type == 2:
        _apply_draw_card(state, pid, action)

    _advance_player(state, pid)


def evaluate(state: SimState) -> List[float]:
    """Evaluate terminal state: route points + ticket bonuses/penalties + longest path.

    Route points are already accumulated in scores during play.
    This adds ticket scoring and longest path bonus.
    """
    final_scores = list(state.scores)

    # Ticket scoring
    for pid in range(state.num_players):
        for ticket, completed in state.tickets[pid].items():
            if completed:
                final_scores[pid] += ticket.points
            else:
                final_scores[pid] -= ticket.points

    # Longest path bonus
    if hasattr(state.config, 'LONGEST_ROUTE_BONUS'):
        bonus = state.config.LONGEST_ROUTE_BONUS
        longest_lengths = []
        for pid in range(state.num_players):
            length = _calculate_longest_path(state, pid)
            longest_lengths.append(length)

        max_length = max(longest_lengths)
        if max_length > 0:
            for pid in range(state.num_players):
                if longest_lengths[pid] == max_length:
                    final_scores[pid] += bonus

    return final_scores


# --- Internal helpers ---

def _wilds_needed(hand: Dict[str, int], color: str, weight: int) -> Optional[int]:
    """Return number of wilds needed, or None if not affordable."""
    color_count = hand.get(color, 0)
    wild_count = hand.get('wild', 0)
    if color_count + wild_count < weight:
        return None
    return max(0, weight - color_count)


def _validate_route(state: SimState, player_id: int, ri: RouteInfo) -> bool:
    """Check if player can claim this route (ownership rules)."""
    sibling_link_ids = state.route_ids_by_route_id.get(ri.route_id, [])
    owners = [state.route_owners.get(lid) for lid in sibling_link_ids]

    if not any(o is not None for o in owners):
        return True  # All links unclaimed

    if player_id in owners:
        return False  # Already claimed one link

    if all(o is not None for o in owners):
        return False  # All links claimed

    # In restricted games with <= 3 players, can't claim parallel routes
    if state.wild_card_restriction and state.num_players <= 3:
        return False

    return True


def _apply_claim_route(state: SimState, pid: int, action: Action) -> None:
    """Claim a route: update board, hand, score, trains, check tickets."""
    ri = _get_route_info(state, action.link_id)
    if ri is None:
        return

    weight = ri.weight
    color = action.color
    wilds = min(action.wilds, state.hands[pid].get('wild', 0))
    color_cards = weight - wilds

    # Remove cards from hand
    state.hands[pid][color] = state.hands[pid].get(color, 0) - color_cards
    state.hands[pid]['wild'] = state.hands[pid].get('wild', 0) - wilds

    # Add to discard
    state.discard_pile.extend([color] * color_cards)
    state.discard_pile.extend(['wild'] * wilds)

    # Claim route
    state.route_owners[action.link_id] = pid

    # Update trains and score
    state.trains[pid] -= weight
    state.scores[pid] += state.route_values.get(weight, 0)

    # Check ticket completion
    _check_completed_tickets(state, pid)


def _apply_draw_tickets(state: SimState, pid: int) -> None:
    """Draw tickets: take up to 3, keep at least 1 (simplified: keep all)."""
    num_draw = min(3, len(state.ticket_deck_remaining))
    if num_draw == 0:
        return

    drawn = []
    for _ in range(num_draw):
        if state.ticket_deck_remaining:
            drawn.append(state.ticket_deck_remaining.pop(0))

    # Simplified: keep all drawn tickets (in rollouts, heuristic can override)
    for ticket in drawn:
        completed = _is_ticket_completed(state, pid, ticket)
        state.tickets[pid][ticket] = completed


def _apply_draw_card(state: SimState, pid: int, action: Action) -> None:
    """Draw cards for a turn (2 cards, or 1 if picking a face-up wild).

    Mirrors the real game's draw_train_cards(2) logic:
    - First card uses the action's choice (face-up or draw pile)
    - If first card is a face-up wild, turn ends (only 1 card drawn)
    - Otherwise, second card drawn from draw pile
    """
    choice = action.card_choice
    drew_wild_face_up = False

    # First card
    if choice == 5:  # Draw pile
        card = _draw_from_pile(state)
        if card is not None:
            state.hands[pid][card] = state.hands[pid].get(card, 0) + 1
    elif 0 <= choice < 5:
        card = state.face_up_cards[choice]
        if card is not None:
            state.face_up_cards[choice] = None
            state.hands[pid][card] = state.hands[pid].get(card, 0) + 1
            if card == 'wild' and state.wild_card_restriction:
                drew_wild_face_up = True
            _fill_face_up(state)

    # Second card (from draw pile, unless first was a face-up wild)
    if not drew_wild_face_up:
        card2 = _draw_from_pile(state)
        if card2 is not None:
            state.hands[pid][card2] = state.hands[pid].get(card2, 0) + 1


def _draw_from_pile(state: SimState) -> Optional[str]:
    """Draw a card from the draw pile, reshuffling discard if needed."""
    if not state.draw_pile:
        if state.discard_pile:
            state.draw_pile = state.discard_pile
            random.shuffle(state.draw_pile)
            state.discard_pile = []
        else:
            return None
    return state.draw_pile.pop() if state.draw_pile else None


def _fill_face_up(state: SimState) -> None:
    """Refill face-up card display from draw pile."""
    max_wild = 3
    tries = 0
    while tries < 5:
        for i in range(5):
            if state.face_up_cards[i] is None:
                card = _draw_from_pile(state)
                if card is None:
                    return
                state.face_up_cards[i] = card

        wild_count = sum(1 for c in state.face_up_cards if c == 'wild')
        if wild_count >= max_wild:
            state.discard_pile.extend(c for c in state.face_up_cards if c is not None)
            state.face_up_cards = [None] * 5
            tries += 1
        else:
            break


def _advance_player(state: SimState, acted_pid: int) -> None:
    """Advance to next player, handle last-round and finished transitions."""
    if state.game_phase == 'running':
        if state.trains[acted_pid] <= state.min_train_figures:
            state.game_phase = 'last_round'
            state.last_player = acted_pid
        state.current_player = (state.current_player + 1) % state.num_players
    elif state.game_phase == 'last_round':
        next_pid = (state.current_player + 1) % state.num_players
        if next_pid == state.last_player:
            state.game_phase = 'finished'
        state.current_player = next_pid


def _get_route_info(state: SimState, link_id: int) -> Optional[RouteInfo]:
    """Find RouteInfo by link_id."""
    for ri in state.route_info:
        if ri.link_id == link_id:
            return ri
    return None


def _check_completed_tickets(state: SimState, pid: int) -> None:
    """Check and update ticket completion status for a player."""
    for ticket in state.tickets[pid]:
        if not state.tickets[pid][ticket]:
            if _is_ticket_completed(state, pid, ticket):
                state.tickets[pid][ticket] = True


def _is_ticket_completed(state: SimState, pid: int, ticket: Ticket) -> bool:
    """Check if player has a path between ticket cities using claimed routes.

    Uses BFS on claimed routes — no NetworkX dependency.
    """
    # Build adjacency from player's claimed routes
    adj: Dict[str, List[str]] = defaultdict(list)
    for ri in state.route_info:
        if state.route_owners.get(ri.link_id) == pid:
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


def _calculate_longest_path(state: SimState, pid: int) -> int:
    """Calculate longest continuous path for a player using DFS."""
    # Build adjacency with edge weights
    adj: Dict[str, List[Tuple[str, int, int]]] = defaultdict(list)  # city -> [(city, weight, link_id)]
    for ri in state.route_info:
        if state.route_owners.get(ri.link_id) == pid:
            adj[ri.city_from].append((ri.city_to, ri.weight, ri.link_id))
            adj[ri.city_to].append((ri.city_from, ri.weight, ri.link_id))

    if not adj:
        return 0

    def dfs(node: str, visited_edges: set) -> int:
        max_length = 0
        for neighbor, weight, lid in adj[node]:
            if lid in visited_edges:
                continue
            visited_edges.add(lid)
            max_length = max(max_length, weight + dfs(neighbor, visited_edges))
            visited_edges.remove(lid)
        return max_length

    longest = 0
    for node in adj:
        longest = max(longest, dfs(node, set()))
    return longest
