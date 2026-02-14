"""AlphaZero MCTS: NN-guided tree search with PUCT and determinization.

Replaces rollouts with neural network evaluation. Each iteration:
1. Determinize hidden state
2. Select via PUCT (using NN policy priors)
3. Expand leaf and evaluate with NN value head
4. Backpropagate value
"""
import math
import random
from typing import List, Optional, Dict, Tuple, TYPE_CHECKING

import torch
import torch.nn.functional as F

from mcts.sim_game import (
    SimState, Action, clone, from_game,
    get_legal_actions, apply_action, is_terminal, evaluate,
)
from mcts.determinize import determinize
from alphazero.encoding import (
    encode_state, action_to_index, get_action_mask, ACTION_SPACE_SIZE,
)

if TYPE_CHECKING:
    from alphazero.network import AlphaZeroNet
    from game.core import Game


class AZNode:
    """Node in the AlphaZero MCTS tree."""
    __slots__ = ['action', 'parent', 'children', 'visits', 'total_value', 'prior']

    def __init__(self, action: Optional[Action] = None, parent: Optional['AZNode'] = None,
                 prior: float = 0.0):
        self.action = action
        self.parent = parent
        self.children: List[AZNode] = []
        self.visits = 0
        self.total_value = 0.0
        self.prior = prior


class AlphaZeroMCTS:
    """NN-guided MCTS with determinization for imperfect information.

    Args:
        network: Trained AlphaZeroNet (or random for initial self-play).
        iterations: Number of MCTS iterations per search.
        c_puct: PUCT exploration constant.
        device: torch device for NN inference.
        dirichlet_alpha: Dirichlet noise parameter for root exploration.
        dirichlet_epsilon: Weight of Dirichlet noise at root.
        temperature: Action selection temperature (1.0 = proportional, 0 = greedy).
    """

    def __init__(
        self,
        network: 'AlphaZeroNet',
        iterations: int = 200,
        c_puct: float = 1.5,
        device: str = 'cpu',
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
    ):
        self.network = network
        self.iterations = iterations
        self.c_puct = c_puct
        self.device = device
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

    @torch.no_grad()
    def search(self, game: 'Game', player_id: int,
               temperature: float = 0.0) -> Tuple[Action, Dict[Action, float]]:
        """Run AlphaZero MCTS and return best action + visit distribution.

        Args:
            game: Live Game object (read-only).
            player_id: Player whose perspective to search from.
            temperature: 0 = greedy, 1.0 = proportional to visits.

        Returns:
            (best_action, policy_dict) where policy_dict maps Action → visit fraction.
        """
        root_state = from_game(game)
        return self.search_from_state(root_state, player_id, temperature)

    @torch.no_grad()
    def search_from_state(self, root_state: SimState, player_id: int,
                          temperature: float = 0.0) -> Tuple[Action, Dict[Action, float]]:
        """Run search directly from a SimState (used by self-play)."""
        self.network.eval()
        root = AZNode()

        # Get root legal actions and initialize with NN priors
        root_legal = get_legal_actions(root_state)
        if not root_legal:
            fallback = Action(action_type=2, card_choice=5)
            return fallback, {fallback: 1.0}

        root_priors = self._get_priors(root_state, player_id, root_legal)

        # Add Dirichlet noise to root priors for exploration
        if self.dirichlet_epsilon > 0 and len(root_legal) > 0:
            noise = torch.distributions.Dirichlet(
                torch.full((len(root_legal),), self.dirichlet_alpha)
            ).sample()
            for i, action in enumerate(root_legal):
                noisy_prior = ((1 - self.dirichlet_epsilon) * root_priors[i]
                               + self.dirichlet_epsilon * noise[i].item())
                child = AZNode(action=action, parent=root, prior=noisy_prior)
                root.children.append(child)
        else:
            for i, action in enumerate(root_legal):
                child = AZNode(action=action, parent=root, prior=root_priors[i])
                root.children.append(child)

        # Main MCTS loop
        for _ in range(self.iterations):
            # 1. Determinize
            det_state = determinize(root_state, player_id)

            # 2. Select
            node = root
            state = clone(det_state)
            det_legal = set(get_legal_actions(state))

            while node.children:
                # Filter to children legal in this determinization
                legal_children = [c for c in node.children if c.action in det_legal]
                if not legal_children:
                    break

                # PUCT select
                node = self._select_child(legal_children)
                apply_action(state, node.action)

                if is_terminal(state):
                    break

                # If this node hasn't been expanded yet, expand it
                if not node.children:
                    child_legal = get_legal_actions(state)
                    if child_legal:
                        priors = self._get_priors(state, player_id, child_legal)
                        for i, action in enumerate(child_legal):
                            child = AZNode(action=action, parent=node, prior=priors[i])
                            node.children.append(child)
                    break

                det_legal = set(get_legal_actions(state))

            # 3. Evaluate leaf
            if is_terminal(state):
                scores = evaluate(state)
                my_score = scores[player_id]
                opp_score = max(s for i, s in enumerate(scores) if i != player_id)
                diff = my_score - opp_score
                value = math.tanh(diff / 30.0)
            else:
                state_tensor = encode_state(state, player_id).unsqueeze(0).to(self.device)
                _, value_tensor = self.network(state_tensor)
                value = value_tensor.item()

            # 4. Backpropagate
            while node is not None:
                node.visits += 1
                node.total_value += value
                node = node.parent

        # Build visit distribution
        if not root.children:
            return random.choice(root_legal), {a: 1.0 / len(root_legal) for a in root_legal}

        visit_counts = {c.action: c.visits for c in root.children}
        total = sum(visit_counts.values())

        if total == 0:
            action = random.choice(root_legal)
            return action, {a: 1.0 / len(root_legal) for a in root_legal}

        # Select action based on temperature
        if temperature <= 0.01:
            # Greedy: pick most visited
            best = max(root.children, key=lambda c: c.visits)
            action = best.action
        else:
            # Proportional to visits^(1/temperature)
            visits = torch.tensor([c.visits for c in root.children], dtype=torch.float32)
            if temperature != 1.0:
                visits = visits ** (1.0 / temperature)
            probs = visits / visits.sum()
            idx = torch.multinomial(probs, 1).item()
            action = root.children[idx].action

        policy = {a: v / total for a, v in visit_counts.items()}
        return action, policy

    def _select_child(self, children: List[AZNode]) -> AZNode:
        """PUCT selection among children."""
        parent_visits = sum(c.visits for c in children)
        sqrt_parent = math.sqrt(max(parent_visits, 1))

        def puct_score(child: AZNode) -> float:
            if child.visits == 0:
                q = 0.0
            else:
                q = child.total_value / child.visits
            u = self.c_puct * child.prior * sqrt_parent / (1 + child.visits)
            return q + u

        return max(children, key=puct_score)

    def _get_priors(self, state: SimState, player_id: int,
                    legal_actions: List[Action]) -> List[float]:
        """Get NN policy priors for legal actions."""
        state_tensor = encode_state(state, player_id).unsqueeze(0).to(self.device)
        policy_logits, _ = self.network(state_tensor)

        mask = get_action_mask(legal_actions).to(self.device)
        masked_logits = policy_logits.squeeze(0).masked_fill(~mask, float('-inf'))
        probs = F.softmax(masked_logits, dim=0)

        return [probs[action_to_index(a)].item() for a in legal_actions]
