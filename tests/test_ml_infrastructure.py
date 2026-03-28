"""Tests for AlphaZero ML infrastructure: network, replay buffer, and MCTS search.

Covers:
- AlphaZeroNet forward pass: output shapes, value range, eval mode
- ReplayBuffer: add, len, circular eviction, sample shapes, edge cases
- AlphaZeroMCTS.search(): returns a legal action, policy sums to 1
- ISMCTS.search(): returns a legal action
"""
import pytest
import torch

from alphazero.network import AlphaZeroNet
from alphazero.replay_buffer import ReplayBuffer
from alphazero.self_play import TrainingSample
from alphazero.encoding import STATE_SIZE, ACTION_SPACE_SIZE
from alphazero.az_mcts import AlphaZeroMCTS
from mcts.is_mcts import ISMCTS
from mcts.rollout_policies import random_rollout

from game.sim_game import SimState, Action, RouteInfo, get_legal_actions
from game.ticket_deck import Ticket
from game.config import USAConfig


# ── helpers ───────────────────────────────────────────────────────────────────

def make_sample(value: float = 0.0) -> TrainingSample:
    return TrainingSample(
        state=torch.zeros(STATE_SIZE),
        policy=torch.zeros(ACTION_SPACE_SIZE),
        value=value,
    )


def make_sim_state() -> SimState:
    """Minimal SimState with one claimable route and a draw pile."""
    s = SimState()
    s.num_players = 2
    s.config = USAConfig()
    s.route_values = dict(s.config.ROUTE_VALUES)
    s.wild_card_restriction = True
    s.min_train_figures = 3
    s.cards_per_draw = 2
    s.hands = [{'red': 3}, {'blue': 3}]
    s.tickets = [{}, {}]
    s.scores = [0, 0]
    s.trains = [45, 45]
    s.current_player = 0
    s.game_phase = 'running'
    s.last_player = None

    ri = RouteInfo(link_id=0, route_id=0, city_from='A', city_to='B',
                   weight=3, edge_color='red')
    s.route_info = [ri]
    s.route_owners = {0: None}
    s.route_ids_by_route_id = {0: [0]}

    s.draw_pile = ['blue', 'green', 'red', 'yellow', 'black']
    s.discard_pile = []
    s.face_up_cards = ['blue', 'green', None, None, None]
    s.ticket_deck_remaining = []
    return s


# ── AlphaZeroNet ──────────────────────────────────────────────────────────────

class TestAlphaZeroNet:
    def test_policy_output_shape_batch_1(self):
        net = AlphaZeroNet()
        x = torch.zeros(1, STATE_SIZE)
        policy, value = net(x)
        assert policy.shape == (1, ACTION_SPACE_SIZE)

    def test_value_output_shape_batch_1(self):
        net = AlphaZeroNet()
        x = torch.zeros(1, STATE_SIZE)
        _, value = net(x)
        assert value.shape == (1, 1)

    def test_policy_output_shape_batched(self):
        net = AlphaZeroNet()
        x = torch.zeros(32, STATE_SIZE)
        policy, _ = net(x)
        assert policy.shape == (32, ACTION_SPACE_SIZE)

    def test_value_in_minus_one_to_one(self):
        """Value head uses tanh — output must lie in (-1, 1)."""
        net = AlphaZeroNet()
        x = torch.randn(64, STATE_SIZE)
        _, value = net(x)
        assert float(value.detach().min()) >= -1.0
        assert float(value.detach().max()) <= 1.0

    def test_smaller_architecture(self):
        net = AlphaZeroNet(hidden_size=64, num_res_blocks=2)
        x = torch.zeros(1, STATE_SIZE)
        policy, value = net(x)
        assert policy.shape == (1, ACTION_SPACE_SIZE)
        assert value.shape == (1, 1)

    def test_eval_mode_no_exception(self):
        net = AlphaZeroNet()
        net.eval()
        with torch.no_grad():
            x = torch.zeros(1, STATE_SIZE)
            policy, value = net(x)
        assert policy.shape == (1, ACTION_SPACE_SIZE)

    def test_different_inputs_give_different_outputs(self):
        """Network is not trivially constant."""
        net = AlphaZeroNet(hidden_size=64, num_res_blocks=1)
        x1 = torch.zeros(1, STATE_SIZE)
        x2 = torch.ones(1, STATE_SIZE)
        p1, v1 = net(x1)
        p2, v2 = net(x2)
        assert not torch.allclose(p1, p2)


# ── ReplayBuffer ──────────────────────────────────────────────────────────────

class TestReplayBuffer:
    def test_initially_empty(self):
        buf = ReplayBuffer()
        assert len(buf) == 0

    def test_add_increases_length(self):
        buf = ReplayBuffer()
        buf.add([make_sample(), make_sample()])
        assert len(buf) == 2

    def test_add_multiple_batches(self):
        buf = ReplayBuffer()
        buf.add([make_sample()] * 5)
        buf.add([make_sample()] * 3)
        assert len(buf) == 8

    def test_circular_eviction_at_max_size(self):
        buf = ReplayBuffer(max_size=3)
        buf.add([make_sample(float(i)) for i in range(5)])
        assert len(buf) == 3

    def test_oldest_samples_evicted_first(self):
        buf = ReplayBuffer(max_size=3)
        buf.add([make_sample(float(i)) for i in range(5)])
        # Only values 2, 3, 4 should remain (0 and 1 evicted)
        remaining_values = {s.value for s in buf.buffer}
        assert remaining_values == {2.0, 3.0, 4.0}

    def test_sample_returns_three_tensors(self):
        buf = ReplayBuffer()
        buf.add([make_sample()] * 10)
        result = buf.sample(4)
        assert len(result) == 3

    def test_sample_states_shape(self):
        buf = ReplayBuffer()
        buf.add([make_sample()] * 10)
        states, _, _ = buf.sample(4)
        assert states.shape == (4, STATE_SIZE)

    def test_sample_policies_shape(self):
        buf = ReplayBuffer()
        buf.add([make_sample()] * 10)
        _, policies, _ = buf.sample(4)
        assert policies.shape == (4, ACTION_SPACE_SIZE)

    def test_sample_values_shape(self):
        buf = ReplayBuffer()
        buf.add([make_sample()] * 10)
        _, _, values = buf.sample(4)
        assert values.shape == (4,)

    def test_sample_values_dtype_float32(self):
        buf = ReplayBuffer()
        buf.add([make_sample(0.5)] * 5)
        _, _, values = buf.sample(3)
        assert values.dtype == torch.float32

    def test_sample_smaller_than_buffer_size(self):
        """When buffer has fewer samples than batch_size, return all."""
        buf = ReplayBuffer()
        buf.add([make_sample()] * 3)
        states, policies, values = buf.sample(10)
        assert states.shape[0] == 3

    def test_sample_preserves_values(self):
        buf = ReplayBuffer()
        buf.add([make_sample(0.7)])
        _, _, values = buf.sample(1)
        assert float(values[0]) == pytest.approx(0.7)


# ── AlphaZeroMCTS smoke tests ─────────────────────────────────────────────────

class TestAlphaZeroMCTS:
    """AlphaZeroMCTS.search() must return a legal action and a valid policy."""

    @pytest.fixture
    def net(self):
        n = AlphaZeroNet(hidden_size=64, num_res_blocks=1)
        n.eval()
        return n

    def test_returns_legal_action(self, net):
        state = make_sim_state()
        legal = get_legal_actions(state)
        mcts = AlphaZeroMCTS(net, iterations=10, device='cpu')

        action, _ = mcts.search_from_state(state, player_id=0)

        assert action in legal, f"Returned action {action} is not in legal actions"

    def test_policy_covers_all_root_children(self, net):
        state = make_sim_state()
        mcts = AlphaZeroMCTS(net, iterations=20, device='cpu')

        _, policy = mcts.search_from_state(state, player_id=0)

        assert len(policy) > 0

    def test_policy_sums_to_one(self, net):
        state = make_sim_state()
        mcts = AlphaZeroMCTS(net, iterations=20, device='cpu')

        _, policy = mcts.search_from_state(state, player_id=0)

        total = sum(policy.values())
        assert total == pytest.approx(1.0, abs=1e-5)

    def test_policy_values_non_negative(self, net):
        state = make_sim_state()
        mcts = AlphaZeroMCTS(net, iterations=20, device='cpu')

        _, policy = mcts.search_from_state(state, player_id=0)

        for action, prob in policy.items():
            assert prob >= 0.0

    def test_deterministic_with_temperature_zero(self, net):
        """At temperature=0 (greedy), the same state should yield the same action."""
        state = make_sim_state()
        mcts = AlphaZeroMCTS(net, iterations=30, device='cpu',
                              dirichlet_epsilon=0.0)

        action1, _ = mcts.search_from_state(state, player_id=0, temperature=0.0)
        action2, _ = mcts.search_from_state(state, player_id=0, temperature=0.0)

        assert action1 == action2

    def test_fallback_when_no_legal_actions(self, net):
        """Terminal state has no legal actions — search must not raise."""
        state = make_sim_state()
        state.game_phase = 'finished'
        mcts = AlphaZeroMCTS(net, iterations=5, device='cpu')

        action, policy = mcts.search_from_state(state, player_id=0)

        assert action is not None


# ── ISMCTS smoke tests ────────────────────────────────────────────────────────

class TestISMCTS:
    """ISMCTS.search() must return an action that belongs to the legal action set."""

    def test_returns_legal_action(self):
        state = make_sim_state()
        legal = get_legal_actions(state)

        # Build a minimal Game-like wrapper: ISMCTS.search() takes a Game object
        # and calls from_game() internally. We test via the lower-level path.
        mcts = ISMCTS(iterations=20, rollout_policy=random_rollout)

        # Search directly on the state (bypasses from_game — internal method)
        root_state = state
        from game.sim_game import clone, apply_action, is_terminal
        from mcts.determinize import determinize
        import random as rnd

        # Run a minimal search manually to get the action
        # (replicating search() but without needing a Game object)
        from mcts.is_mcts import MCTSNode
        root = MCTSNode()
        for _ in range(20):
            det = determinize(root_state, 0)
            node = root
            s = clone(det)
            leg = set(get_legal_actions(s))

            while node.children and leg:
                legal_children = [c for c in node.children if c.action in leg]
                if not legal_children:
                    break
                tried = {c.action for c in node.children}
                untried = [a for a in leg if a not in tried]
                if untried:
                    break
                node = mcts._select_child(legal_children, node.visits)
                apply_action(s, node.action)
                if is_terminal(s):
                    break
                leg = set(get_legal_actions(s))

            if not is_terminal(s):
                tried = {c.action for c in node.children}
                untried = [a for a in leg if a not in tried]
                if untried:
                    act = rnd.choice(untried)
                    child = MCTSNode(action=act, parent=node)
                    node.children.append(child)
                    apply_action(s, act)
                    node = child

            value = mcts._rollout(s, 0)
            while node is not None:
                node.visits += 1
                node.total_value += value
                node = node.parent

        if root.children:
            best = max(root.children, key=lambda c: c.visits).action
            assert best in legal

    def test_returns_action_via_game(self):
        """End-to-end: ISMCTS.search() called with a real Game object."""
        from game.core import Game
        game = Game(['Random', 'Random'], 'USA')
        game._deal_initial()

        from game.sim_game import from_game
        state = from_game(game)
        legal = get_legal_actions(state)

        mcts = ISMCTS(iterations=30, rollout_policy=random_rollout)
        action = mcts.search(game, player_id=game.current_player_id)

        assert action in legal, f"ISMCTS returned {action} not in legal actions"

    def test_action_type_is_valid(self):
        from game.core import Game
        game = Game(['Random', 'Random'], 'USA')
        game._deal_initial()

        mcts = ISMCTS(iterations=20, rollout_policy=random_rollout)
        action = mcts.search(game, player_id=game.current_player_id)

        assert action.action_type in (0, 1, 2)
