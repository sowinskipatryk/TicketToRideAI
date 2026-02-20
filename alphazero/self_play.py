"""Self-play game generation for AlphaZero training.

Plays games using AlphaZeroMCTS, recording training samples:
(state_tensor, policy_tensor, value_target).
"""
import math
from dataclasses import dataclass
from typing import List, Tuple

import torch

from game.core import Game
from mcts.sim_game import SimState, Action, from_game, get_legal_actions, apply_action, is_terminal, evaluate
from alphazero.encoding import encode_state, policy_dict_to_tensor, ENCODING_GAME_VERSION
from alphazero.az_mcts import AlphaZeroMCTS


@dataclass
class TrainingSample:
    state: torch.Tensor       # (STATE_SIZE,)
    policy: torch.Tensor      # (ACTION_SPACE_SIZE,)
    value: float              # Target value in [-1, 1]


def create_initial_state(version: str = ENCODING_GAME_VERSION, num_players: int = 2) -> SimState:
    """Create a fresh SimState for self-play by initializing a Game and converting it."""
    game = Game(['Random'] * num_players, version)
    game._deal_initial()
    return from_game(game)


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
