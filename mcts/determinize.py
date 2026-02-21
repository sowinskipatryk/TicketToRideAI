"""Determinization for IS-MCTS.

Creates plausible complete game states from the perspective of a given player
by randomly sampling the hidden information (opponent hands, draw pile order,
opponent tickets, ticket deck order).
"""
import random
from collections import defaultdict
from typing import Dict, List, Set

from game.ticket_deck import Ticket
from game.sim_game import SimState, clone


def determinize(state: SimState, perspective: int) -> SimState:
    """Create a determinized copy where hidden info is randomly sampled.

    What the perspective player knows:
    - Their own hand (exact cards)
    - Their own tickets
    - Face-up cards
    - Discard pile contents
    - Opponents' hand SIZES (but not card identities)
    - Opponents' ticket COUNTS (but not which tickets)
    - Board state (all claimed routes)

    What we randomize:
    1. Opponents' hand contents (maintaining correct sizes)
    2. Draw pile order
    3. Opponent tickets (maintaining correct counts)
    4. Ticket deck order
    """
    det = clone(state)

    # --- Card determinization ---
    # Build the pool of cards whose location is unknown to perspective player
    pool = _build_unknown_card_pool(state, perspective)
    random.shuffle(pool)

    # Distribute cards to opponent hands (maintaining sizes), rest to draw pile
    offset = 0
    for pid in range(state.num_players):
        if pid == perspective:
            continue
        hand_size = sum(state.hands[pid].values())
        sampled = pool[offset:offset + hand_size]
        offset += hand_size
        det.hands[pid] = _cards_to_hand(sampled)

    # Remaining cards form the draw pile (shuffled)
    det.draw_pile = pool[offset:]
    random.shuffle(det.draw_pile)

    # --- Ticket determinization ---
    # Shuffle ticket deck (we don't know its order)
    random.shuffle(det.ticket_deck_remaining)

    # Sample opponent tickets
    _sample_opponent_tickets(det, perspective)

    return det


def _build_unknown_card_pool(state: SimState, perspective: int) -> List[str]:
    """Build list of cards whose location is unknown to the perspective player.

    Total cards in the game = (NUM_TRAIN_CARDS_PER_COLOR * num_colors) + NUM_WILD_CARDS
    Known locations: perspective's hand + face_up + discard_pile
    Unknown: opponent hands + draw pile (we pool these and redistribute)
    """
    # Count all cards in opponent hands + draw pile
    pool = []

    # Add all opponent hand cards
    for pid in range(state.num_players):
        if pid == perspective:
            continue
        for color, count in state.hands[pid].items():
            pool.extend([color] * count)

    # Add draw pile
    pool.extend(state.draw_pile)

    return pool


def _cards_to_hand(cards: List[str]) -> Dict[str, int]:
    """Convert a list of card strings to a hand dict."""
    hand: Dict[str, int] = defaultdict(int)
    for card in cards:
        hand[card] += 1
    return dict(hand)


def _sample_opponent_tickets(state: SimState, perspective: int) -> None:
    """Re-sample opponent tickets while maintaining counts.

    We know how many tickets each opponent has but not which ones.
    Sample uniformly from tickets not held by the perspective player.
    """
    # Collect all tickets held by perspective player
    my_tickets: Set[Ticket] = set(state.tickets[perspective].keys())

    # Collect all tickets currently assigned to any player
    all_held_tickets: Set[Ticket] = set()
    for pid in range(state.num_players):
        all_held_tickets.update(state.tickets[pid].keys())

    # Build pool of available tickets for sampling
    # Include: tickets in the deck + tickets held by opponents
    available = list(state.ticket_deck_remaining)
    for pid in range(state.num_players):
        if pid == perspective:
            continue
        available.extend(state.tickets[pid].keys())

    # Remove duplicates (a ticket can only appear once)
    available_set = set(available) - my_tickets
    available = list(available_set)
    random.shuffle(available)

    # Assign to each opponent
    offset = 0
    for pid in range(state.num_players):
        if pid == perspective:
            continue
        num_tickets = len(state.tickets[pid])
        sampled = available[offset:offset + num_tickets]
        offset += num_tickets

        new_tickets = {}
        for ticket in sampled:
            # Check if this ticket is already completed by this player
            # (we can see their routes, so we can check completion)
            from game.sim_game import _is_ticket_completed
            new_tickets[ticket] = _is_ticket_completed(state, pid, ticket)
        state.tickets[pid] = new_tickets

    # Rebuild ticket deck with remaining available tickets
    remaining = available[offset:]
    # Also add back the original opponent tickets that weren't re-sampled
    state.ticket_deck_remaining = remaining
    random.shuffle(state.ticket_deck_remaining)
