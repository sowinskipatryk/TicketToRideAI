"""Self-play game generation for AlphaZero training.

Plays games using AlphaZeroMCTS, recording training samples:
(state_tensor, policy_tensor, value_target).
"""
import math
import random
from dataclasses import dataclass
from typing import List, Tuple

import torch

from game.data_loader import load_routes, load_tickets
from game.config import ConfigFactory
from game.ticket_deck import Ticket
from mcts.sim_game import (
    SimState, RouteInfo, Action,
    get_legal_actions, apply_action, is_terminal, evaluate,
    _fill_face_up, _is_ticket_completed,
)
from alphazero.encoding import encode_state, policy_dict_to_tensor, COLOR_TO_IDX
from alphazero.az_mcts import AlphaZeroMCTS


@dataclass
class TrainingSample:
    state: torch.Tensor       # (STATE_SIZE,)
    policy: torch.Tensor      # (ACTION_SPACE_SIZE,)
    value: float              # Target value in [-1, 1]


def create_initial_state(version: str = 'USA', num_players: int = 2) -> SimState:
    """Create a fresh SimState for self-play without the full Game object.

    Replicates Game.__init__ + initial dealing logic using only data files.
    """
    config = ConfigFactory.create(version)
    state = SimState()
    state.num_players = num_players
    state.config = config
    state.route_values = dict(config.ROUTE_VALUES)
    state.wild_card_restriction = getattr(config, 'WILD_CARD_RESTRICTION', True)
    state.min_train_figures = config.MIN_TRAIN_FIGURES
    state.cards_per_draw = config.NUM_TRAIN_CARDS_DEALT

    # Build routes from data
    routes_data = load_routes(version)
    route_info_list = []
    route_ids_map = {}
    link_id = 0
    for route_id, route in enumerate(routes_data):
        for color in route['colors']:
            ri = RouteInfo(
                link_id=link_id,
                route_id=route_id,
                city_from=route['from'],
                city_to=route['to'],
                weight=route['length'],
                edge_color=color,
            )
            route_info_list.append(ri)
            state.route_owners[link_id] = None
            route_ids_map.setdefault(route_id, []).append(link_id)
            link_id += 1

    state.route_info = route_info_list
    state.route_ids_by_route_id = route_ids_map

    # Build card deck
    train_colors = config.TRAIN_COLORS
    cards = []
    for color in train_colors:
        cards.extend([color] * config.NUM_TRAIN_CARDS_PER_COLOR)
    cards.extend(['wild'] * config.NUM_WILD_CARDS)
    random.shuffle(cards)
    state.draw_pile = cards
    state.discard_pile = []
    state.face_up_cards = [None] * 5

    # Build ticket deck
    tickets_data = load_tickets(version)
    ticket_list = [Ticket(t['from'], t['to'], t['points']) for t in tickets_data]
    random.shuffle(ticket_list)
    state.ticket_deck_remaining = ticket_list

    # Initialize per-player state
    state.hands = [{} for _ in range(num_players)]
    state.tickets = [{} for _ in range(num_players)]
    state.scores = [0] * num_players
    state.trains = [config.NUM_TRAIN_FIGURES] * num_players

    # Deal initial cards (4 per player from draw pile)
    for pid in range(num_players):
        for _ in range(config.NUM_TRAIN_CARDS_DEALT_INIT):
            if state.draw_pile:
                card = state.draw_pile.pop()
                state.hands[pid][card] = state.hands[pid].get(card, 0) + 1

    # Set up face-up cards
    _fill_face_up(state)

    # Deal initial tickets (3 per player, keep all for simplicity)
    for pid in range(num_players):
        num_deal = min(config.NUM_TICKETS_DEALT_INIT, len(state.ticket_deck_remaining))
        for _ in range(num_deal):
            if state.ticket_deck_remaining:
                ticket = state.ticket_deck_remaining.pop(0)
                completed = _is_ticket_completed(state, pid, ticket)
                state.tickets[pid][ticket] = completed

    # Game flow
    state.current_player = random.randrange(num_players)
    state.game_phase = 'running'
    state.last_player = None

    return state


def self_play_game(
    network,
    device: str = 'cpu',
    mcts_iterations: int = 100,
    temperature_threshold: int = 15,
) -> List[TrainingSample]:
    """Play one self-play game, returning training samples.

    Args:
        network: AlphaZeroNet to use for MCTS evaluation.
        device: Torch device.
        mcts_iterations: MCTS iterations per move.
        temperature_threshold: Use temperature=1.0 for first N moves, then 0.1.

    Returns:
        List of TrainingSample with value targets assigned from game outcome.
    """
    state = create_initial_state()
    az_mcts = AlphaZeroMCTS(
        network, iterations=mcts_iterations, c_puct=1.5,
        device=device, dirichlet_alpha=0.3, dirichlet_epsilon=0.25,
    )

    history: List[Tuple[torch.Tensor, torch.Tensor, int]] = []
    move_count = 0

    while not is_terminal(state):
        pid = state.current_player
        temp = 1.0 if move_count < temperature_threshold else 0.1

        # Encode state
        state_tensor = encode_state(state, pid)

        # Run MCTS
        action, policy = az_mcts.search_from_state(state, pid, temperature=temp)

        # Record sample
        policy_tensor = policy_dict_to_tensor(policy)
        history.append((state_tensor, policy_tensor, pid))

        # Apply action
        apply_action(state, action)
        move_count += 1

        # Safety: cap at 300 moves
        if move_count >= 300:
            break

    # Compute value targets from final scores
    scores = evaluate(state)
    samples = []
    for state_tensor, policy_tensor, pid in history:
        opp = 1 - pid
        diff = scores[pid] - scores[opp]
        # Map to [-1, 1] using tanh scaling
        value_target = math.tanh(diff / 30.0)
        samples.append(TrainingSample(
            state=state_tensor,
            policy=policy_tensor,
            value=value_target,
        ))

    return samples
