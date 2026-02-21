"""Information Set Monte Carlo Tree Search (IS-MCTS).

Uses determinization to handle imperfect information: at each iteration,
sample a plausible complete game state, then run standard MCTS
(select/expand/simulate/backpropagate) on it. Statistics accumulate
across determinizations, finding actions robust to hidden information.
"""
import math
import random
from typing import List, Optional, Callable, TYPE_CHECKING

from game.sim_game import (
    SimState, Action, clone, from_game,
    get_legal_actions, apply_action, is_terminal, evaluate,
)
from mcts.determinize import determinize
from mcts.rollout_policies import random_rollout

if TYPE_CHECKING:
    from game.core import Game


class MCTSNode:
    """A node in the MCTS search tree."""
    __slots__ = ['action', 'parent', 'children', 'visits', 'total_value']

    def __init__(self, action: Optional[Action] = None, parent: Optional['MCTSNode'] = None):
        self.action = action
        self.parent = parent
        self.children: List[MCTSNode] = []
        self.visits = 0
        self.total_value = 0.0


class ISMCTS:
    """Information Set MCTS with determinization.

    Each iteration:
    1. Determinize hidden state (sample opponent hands, draw pile, tickets)
    2. Select: walk tree via UCB1, only following children legal in this determinization
    3. Expand: add one child for an untried legal action
    4. Simulate: rollout to terminal using rollout policy
    5. Backpropagate: update visit counts and values

    Args:
        iterations: Number of MCTS iterations per search call.
        exploration: UCB1 exploration constant (sqrt(2) ≈ 1.41 is standard).
        rollout_policy: Function(state, actions) -> action for simulation phase.
        max_rollout_depth: Maximum moves during rollout before early termination.
    """

    def __init__(
        self,
        iterations: int = 200,
        exploration: float = 1.41,
        rollout_policy: Optional[Callable] = None,
        max_rollout_depth: int = 60,
    ):
        self.iterations = iterations
        self.exploration = exploration
        self.rollout_policy = rollout_policy or random_rollout
        self.max_rollout_depth = max_rollout_depth

    def search(self, game: 'Game', player_id: int) -> Action:
        """Run IS-MCTS and return the best action for player_id.

        Args:
            game: The live Game object (read-only — state is extracted, not modified).
            player_id: The player whose perspective we search from.

        Returns:
            The Action with the most visits (most robust across determinizations).
        """
        root_state = from_game(game)
        root = MCTSNode()

        for _ in range(self.iterations):
            # 1. DETERMINIZE
            det_state = determinize(root_state, player_id)

            # 2. SELECT — walk tree following legal children
            node = root
            state = clone(det_state)
            legal = set(get_legal_actions(state))

            while node.children and legal:
                # Filter to children whose actions are legal in this determinization
                legal_children = [c for c in node.children if c.action in legal]
                if not legal_children:
                    break

                # Check if there are untried legal actions
                tried_actions = {c.action for c in node.children}
                untried = [a for a in legal if a not in tried_actions]
                if untried:
                    break  # Expand instead of selecting

                # UCB1 select among legal children
                node = self._select_child(legal_children)
                apply_action(state, node.action)
                if is_terminal(state):
                    break
                legal = set(get_legal_actions(state))

            # 3. EXPAND — add one untried legal action
            if not is_terminal(state):
                tried_actions = {c.action for c in node.children}
                untried = [a for a in legal if a not in tried_actions]
                if untried:
                    action = random.choice(untried)
                    child = MCTSNode(action=action, parent=node)
                    node.children.append(child)
                    apply_action(state, action)
                    node = child

            # 4. SIMULATE (rollout)
            value = self._rollout(state, player_id)

            # 5. BACKPROPAGATE
            while node is not None:
                node.visits += 1
                node.total_value += value
                node = node.parent

        # Return most-visited child action
        if not root.children:
            # Fallback: no search iterations succeeded, pick random legal action
            fallback_actions = get_legal_actions(root_state)
            if fallback_actions:
                return random.choice(fallback_actions)
            return Action(action_type=2, card_choice=5)  # Draw from pile

        return max(root.children, key=lambda c: c.visits).action

    def _select_child(self, children: List[MCTSNode]) -> MCTSNode:
        """UCB1 selection among children."""
        # Find total visits for parent (sum of children visits works for normalization)
        total_visits = sum(c.visits for c in children)
        log_total = math.log(max(total_visits, 1))

        def ucb1(child: MCTSNode) -> float:
            if child.visits == 0:
                return float('inf')
            exploit = child.total_value / child.visits
            explore = self.exploration * math.sqrt(log_total / child.visits)
            return exploit + explore

        return max(children, key=ucb1)

    def _rollout(self, state: SimState, player_id: int) -> float:
        """Simulate to terminal using rollout policy, return normalized score.

        Uses sigmoid of score difference for smoother gradient than binary win/loss.
        """
        for _ in range(self.max_rollout_depth):
            if is_terminal(state):
                break
            actions = get_legal_actions(state)
            if not actions:
                break
            action = self.rollout_policy(state, actions)
            apply_action(state, action)

        scores = evaluate(state)

        my_score = scores[player_id]
        max_opponent = max(s for i, s in enumerate(scores) if i != player_id)
        diff = my_score - max_opponent
        # Sigmoid: maps diff to (0, 1), with 0 diff -> 0.5
        # Scale of 30 means a 30-point lead gives ~0.73
        return 1.0 / (1.0 + math.exp(-diff / 30.0))
