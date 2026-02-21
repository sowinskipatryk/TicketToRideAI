"""State and action encoding for AlphaZero.

Converts SimState to fixed-size tensors and maps between Action objects
and fixed policy indices. All encoding is from a specific player's perspective.
"""
import torch
from typing import List, Dict, Tuple, Optional

from game.ticket_deck import Ticket
from game.data_loader import load_tickets
from game.sim_game import SimState, Action

# --- Constants ---

NUM_LINKS = 100       # Max route link_ids (USA has exactly 100)
NUM_COLORS = 9        # 8 train colors + wild
NUM_TICKETS = 30      # USA ticket count
NUM_FACE_UP = 5       # Face-up card slots

COLOR_TO_IDX = {
    'red': 0, 'blue': 1, 'green': 2, 'yellow': 3,
    'black': 4, 'white': 5, 'pink': 6, 'orange': 7, 'wild': 8,
}
IDX_TO_COLOR = {v: k for k, v in COLOR_TO_IDX.items()}

# Action space layout
CLAIM_ROUTE_OFFSET = 0                          # 0..899
DRAW_CARD_OFFSET = NUM_LINKS * NUM_COLORS       # 900..905
DRAW_TICKETS_OFFSET = DRAW_CARD_OFFSET + 6      # 906
ACTION_SPACE_SIZE = DRAW_TICKETS_OFFSET + 1      # 907

# Game version this encoding is built for. Must match the version used during training.
# Changing this requires retraining the network from scratch (different STATE_SIZE / NUM_TICKETS).
ENCODING_GAME_VERSION = 'USA'

# Build canonical ticket ordering (sorted by city_from, city_to)
_TICKET_DATA = load_tickets(ENCODING_GAME_VERSION)
CANONICAL_TICKETS: List[Ticket] = sorted(
    [Ticket(t['from'], t['to'], t['points']) for t in _TICKET_DATA],
    key=lambda t: (t.city_from, t.city_to),
)
TICKET_TO_IDX: Dict[Ticket, int] = {t: i for i, t in enumerate(CANONICAL_TICKETS)}

# State feature sizes
_HAND_SIZE = NUM_COLORS           # 9
_OPP_HAND_SIZE = 1                # 1
_FACE_UP_SIZE = NUM_COLORS        # 9
_ROUTE_SIZE = NUM_LINKS * 3       # 300
_TRAINS_SIZE = 2                  # 2 (me + opponent)
_SCORE_SIZE = 2                   # 2
_TICKETS_SIZE = NUM_TICKETS * 2   # 60
_OPP_TICKETS_SIZE = 1             # 1
_DECK_SIZES = 2                   # 2 (ticket deck + draw pile)
_PHASE_SIZE = 3                   # 3

STATE_SIZE = (
    _HAND_SIZE + _OPP_HAND_SIZE + _FACE_UP_SIZE + _ROUTE_SIZE
    + _TRAINS_SIZE + _SCORE_SIZE + _TICKETS_SIZE + _OPP_TICKETS_SIZE
    + _DECK_SIZES + _PHASE_SIZE
)  # 389


def encode_state(state: SimState, perspective: int) -> torch.Tensor:
    """Encode SimState as a 1D float tensor from perspective player's view.

    Args:
        state: The game state to encode.
        perspective: Player ID whose perspective to encode from.

    Returns:
        Tensor of shape (STATE_SIZE,) with values in [0, 1].
    """
    features = []
    opp = 1 - perspective  # Works for 2-player

    # My hand: 9 values (count / 12 per card type)
    for color_idx in range(NUM_COLORS):
        color = IDX_TO_COLOR[color_idx]
        features.append(state.hands[perspective].get(color, 0) / 12.0)

    # Opponent hand size: 1 value
    opp_hand_total = sum(state.hands[opp].values())
    features.append(opp_hand_total / 50.0)

    # Face-up cards: 9 values (count per type / 5)
    face_up_counts = [0] * NUM_COLORS
    for card in state.face_up_cards:
        if card is not None and card in COLOR_TO_IDX:
            face_up_counts[COLOR_TO_IDX[card]] += 1
    for count in face_up_counts:
        features.append(count / 5.0)

    # Route ownership: 100 links × 3 (unclaimed, mine, theirs)
    for link_id in range(NUM_LINKS):
        owner = state.route_owners.get(link_id)
        if owner is None:
            features.extend([1.0, 0.0, 0.0])
        elif owner == perspective:
            features.extend([0.0, 1.0, 0.0])
        else:
            features.extend([0.0, 0.0, 1.0])

    # Trains remaining: 2 values
    features.append(state.trains[perspective] / 45.0)
    features.append(state.trains[opp] / 45.0)

    # Scores: 2 values
    features.append(state.scores[perspective] / 200.0)
    features.append(state.scores[opp] / 200.0)

    # My tickets: 30 × 2 (held, completed)
    for ticket in CANONICAL_TICKETS:
        if ticket in state.tickets[perspective]:
            features.append(1.0)  # held
            features.append(1.0 if state.tickets[perspective][ticket] else 0.0)  # completed
        else:
            features.extend([0.0, 0.0])

    # Opponent ticket count: 1 value
    features.append(len(state.tickets[opp]) / 10.0)

    # Deck sizes: 2 values
    features.append(len(state.ticket_deck_remaining) / 30.0)
    features.append(len(state.draw_pile) / 110.0)

    # Game phase: 3 one-hot
    features.append(1.0 if state.game_phase == 'running' else 0.0)
    features.append(1.0 if state.game_phase == 'last_round' else 0.0)
    features.append(1.0 if state.game_phase == 'finished' else 0.0)

    return torch.tensor(features, dtype=torch.float32)


def action_to_index(action: Action) -> int:
    """Map an Action to a fixed policy index (0..906)."""
    if action.action_type == 0:  # CLAIM_ROUTE
        color_idx = COLOR_TO_IDX.get(action.color, 0)
        return CLAIM_ROUTE_OFFSET + action.link_id * NUM_COLORS + color_idx
    elif action.action_type == 2:  # DRAW_CARDS
        return DRAW_CARD_OFFSET + action.card_choice
    elif action.action_type == 1:  # DRAW_TICKETS
        return DRAW_TICKETS_OFFSET
    raise ValueError(f"Unknown action_type: {action.action_type}")


def index_to_action(index: int, legal_actions: List[Action]) -> Optional[Action]:
    """Map a policy index back to an Action by matching against legal actions.

    Returns the matching legal action, or None if no match found.
    """
    for action in legal_actions:
        if action_to_index(action) == index:
            return action
    return None


def get_action_mask(legal_actions: List[Action]) -> torch.Tensor:
    """Return a boolean mask of shape (ACTION_SPACE_SIZE,) for legal actions."""
    mask = torch.zeros(ACTION_SPACE_SIZE, dtype=torch.bool)
    for action in legal_actions:
        idx = action_to_index(action)
        mask[idx] = True
    return mask


def policy_dict_to_tensor(policy: Dict[Action, float]) -> torch.Tensor:
    """Convert a {Action: visit_fraction} dict to a fixed-size policy tensor."""
    tensor = torch.zeros(ACTION_SPACE_SIZE, dtype=torch.float32)
    for action, prob in policy.items():
        idx = action_to_index(action)
        tensor[idx] = prob
    return tensor
