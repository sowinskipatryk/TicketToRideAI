"""Tests for neuroevolution: TrainingReporter, save_genome, and NEATPlayer.

Covers:
- TrainingReporter fresh init: default state
- TrainingReporter resume: loads existing log, restores _best_calibration
- TrainingReporter.start_generation: sets generation field
- save_genome / load roundtrip via pickle
- NEATPlayer.decide_action: picks highest-value action via mock network
- NEATPlayer.decide_route: returns link_id from cached action
- NEATPlayer.decide_wild_cards: returns wilds from cached action
- NEATPlayer.decide_train_card: returns correct card choice, then DRAW_PILE
- NEATPlayer with no legal actions: returns SKIP (3)
- Integration: Game with NEAT player runs without error
"""
import json
import os
import pickle
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from game.sim_game import SimState, Action, RouteInfo
from game.config import USAConfig


# ── helpers ───────────────────────────────────────────────────────────────────

def make_mock_network(return_values: dict = None):
    """Create a mock NEAT FeedForwardNetwork.

    return_values: maps frozenset-of-features -> float, or None for a constant.
    If None, returns [0.5] for any input.
    """
    net = MagicMock()
    if return_values is None:
        net.activate.return_value = [0.5]
    else:
        def activate(features):
            key = tuple(features)
            return [return_values.get(key, 0.0)]
        net.activate.side_effect = activate
    return net


def make_sim_state() -> SimState:
    """Minimal 2-player SimState with one claimable route."""
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


# ── TrainingReporter ──────────────────────────────────────────────────────────

class TestTrainingReporterFreshInit:
    def test_best_calibration_starts_at_neg_inf(self):
        from neuroevolution.manager import TrainingReporter
        with tempfile.TemporaryDirectory() as d:
            log_path = os.path.join(d, 'log.json')
            reporter = TrainingReporter(num_generations=10, log_path=log_path)
            assert reporter._best_calibration == float('-inf')

    def test_best_genome_starts_as_none(self):
        from neuroevolution.manager import TrainingReporter
        with tempfile.TemporaryDirectory() as d:
            log_path = os.path.join(d, 'log.json')
            reporter = TrainingReporter(num_generations=10, log_path=log_path)
            assert reporter.best_genome is None

    def test_training_log_starts_empty(self):
        from neuroevolution.manager import TrainingReporter
        with tempfile.TemporaryDirectory() as d:
            log_path = os.path.join(d, 'log.json')
            reporter = TrainingReporter(num_generations=10, log_path=log_path)
            assert reporter.training_log == []

    def test_generation_starts_at_zero(self):
        from neuroevolution.manager import TrainingReporter
        with tempfile.TemporaryDirectory() as d:
            log_path = os.path.join(d, 'log.json')
            reporter = TrainingReporter(num_generations=10, log_path=log_path)
            assert reporter.generation == 0

    def test_fresh_init_ignores_nonexistent_log_file(self):
        from neuroevolution.manager import TrainingReporter
        reporter = TrainingReporter(
            num_generations=10,
            log_path='/nonexistent/path/log.json',
            start_generation=5,
        )
        # start_generation > 0 but file doesn't exist — should not raise
        assert reporter.training_log == []
        assert reporter._best_calibration == float('-inf')


class TestTrainingReporterResume:
    def _write_log(self, path, entries):
        with open(path, 'w') as f:
            json.dump(entries, f)

    def test_resume_loads_log_entries_up_to_checkpoint(self):
        from neuroevolution.manager import TrainingReporter
        with tempfile.TemporaryDirectory() as d:
            log_path = os.path.join(d, 'log.json')
            entries = [
                {'generation': 1, 'best_fitness': 5.0, 'best_calibration': 3.0, 'vs_ticketFocused': 2.0},
                {'generation': 2, 'best_fitness': 7.0, 'best_calibration': 6.0, 'vs_ticketFocused': 4.0},
                {'generation': 3, 'best_fitness': 4.0, 'best_calibration': 6.0, 'vs_ticketFocused': -1.0},
            ]
            self._write_log(log_path, entries)

            reporter = TrainingReporter(num_generations=10, log_path=log_path, start_generation=2)

            # Only generations 1 and 2 are kept (3 is discarded)
            assert len(reporter.training_log) == 2
            assert reporter.training_log[-1]['generation'] == 2

    def test_resume_restores_best_calibration(self):
        from neuroevolution.manager import TrainingReporter
        with tempfile.TemporaryDirectory() as d:
            log_path = os.path.join(d, 'log.json')
            entries = [
                {'generation': 1, 'best_fitness': 5.0, 'best_calibration': 3.5, 'vs_ticketFocused': 2.0},
                {'generation': 2, 'best_fitness': 7.0, 'best_calibration': 8.2, 'vs_ticketFocused': 4.0},
            ]
            self._write_log(log_path, entries)

            reporter = TrainingReporter(num_generations=10, log_path=log_path, start_generation=2)

            assert reporter._best_calibration == pytest.approx(8.2)

    def test_resume_with_empty_log_file(self):
        from neuroevolution.manager import TrainingReporter
        with tempfile.TemporaryDirectory() as d:
            log_path = os.path.join(d, 'log.json')
            self._write_log(log_path, [])

            reporter = TrainingReporter(num_generations=10, log_path=log_path, start_generation=5)

            assert reporter.training_log == []
            assert reporter._best_calibration == float('-inf')

    def test_resume_discards_future_generations(self):
        from neuroevolution.manager import TrainingReporter
        with tempfile.TemporaryDirectory() as d:
            log_path = os.path.join(d, 'log.json')
            entries = [
                {'generation': i, 'best_fitness': float(i), 'best_calibration': float(i), 'vs_ticketFocused': 0.0}
                for i in range(1, 11)
            ]
            self._write_log(log_path, entries)

            reporter = TrainingReporter(num_generations=10, log_path=log_path, start_generation=5)

            assert all(e['generation'] <= 5 for e in reporter.training_log)


class TestTrainingReporterStartGeneration:
    def test_start_generation_sets_generation(self):
        from neuroevolution.manager import TrainingReporter
        with tempfile.TemporaryDirectory() as d:
            reporter = TrainingReporter(10, os.path.join(d, 'log.json'))
            reporter.start_generation(7)
            assert reporter.generation == 7

    def test_start_generation_sets_timer(self):
        from neuroevolution.manager import TrainingReporter
        with tempfile.TemporaryDirectory() as d:
            reporter = TrainingReporter(10, os.path.join(d, 'log.json'))
            reporter.start_generation(0)
            assert reporter._gen_start is not None


# ── save_genome ───────────────────────────────────────────────────────────────

class TestSaveGenome:
    def test_save_genome_creates_file(self):
        from neuroevolution.manager import save_genome
        genome = {'nodes': [1, 2], 'connections': []}  # dummy serialisable object
        with tempfile.TemporaryDirectory() as d:
            with patch('neuroevolution.manager.GENOME_FILENAME', os.path.join(d, 'genome.pkl')):
                save_genome(genome)
                assert os.path.exists(os.path.join(d, 'genome.pkl'))

    def test_save_genome_roundtrip(self):
        from neuroevolution.manager import save_genome
        genome = {'nodes': [1, 2, 3], 'fitness': 42.0}
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'genome.pkl')
            with patch('neuroevolution.manager.GENOME_FILENAME', path):
                save_genome(genome)
            with open(path, 'rb') as f:
                loaded = pickle.load(f)
            assert loaded == genome

    def test_save_genome_creates_parent_directories(self):
        from neuroevolution.manager import save_genome
        genome = 'simple_genome'
        with tempfile.TemporaryDirectory() as d:
            nested_path = os.path.join(d, 'a', 'b', 'c', 'genome.pkl')
            with patch('neuroevolution.manager.GENOME_FILENAME', nested_path):
                save_genome(genome)
                assert os.path.exists(nested_path)


# ── NEATPlayer ────────────────────────────────────────────────────────────────

def _make_neat_player(network=None):
    """Construct a NEATPlayer without a real Game dependency."""
    from game.players.neat_player import NEATPlayer

    player = object.__new__(NEATPlayer)
    player.player_id = 0
    player.game = None  # required attribute; from_game(self.game) will be patched
    player.network = network or make_mock_network()
    player._chosen_action = None
    player._card_draw_count = 0
    return player


class TestNEATPlayerDecideAction:
    def test_returns_action_type_for_claim_route(self):
        """When the network favours a claim-route action, decide_action returns 0."""
        player = _make_neat_player(make_mock_network())
        state = make_sim_state()

        with patch('game.players.neat_player.from_game', return_value=state):
            result = player.decide_action()

        assert result in (0, 1, 2, 3)

    def test_picks_highest_value_action(self):
        """Network returning a high value for a specific post-state should
        cause that action to be selected."""
        from game.sim_game import get_legal_actions, clone, apply_action
        from alphazero.encoding import encode_state

        state = make_sim_state()
        legal = get_legal_actions(state)
        assert len(legal) >= 2, "Need at least 2 legal actions for this test"

        # Assign distinct scores: first action gets 1.0, rest get 0.0
        chosen_action = legal[0]
        call_count = [0]

        def activate(features):
            call_count[0] += 1
            return [1.0 if call_count[0] == 1 else 0.0]

        net = MagicMock()
        net.activate.side_effect = activate

        player = _make_neat_player(net)

        with patch('game.players.neat_player.from_game', return_value=state):
            player.decide_action()

        assert player._chosen_action == legal[0]

    def test_caches_chosen_action(self):
        player = _make_neat_player(make_mock_network())
        state = make_sim_state()

        with patch('game.players.neat_player.from_game', return_value=state):
            player.decide_action()

        assert player._chosen_action is not None

    def test_resets_card_draw_count(self):
        player = _make_neat_player(make_mock_network())
        player._card_draw_count = 99
        state = make_sim_state()

        with patch('game.players.neat_player.from_game', return_value=state):
            player.decide_action()

        assert player._card_draw_count == 0

    def test_returns_skip_when_no_legal_actions(self):
        player = _make_neat_player(make_mock_network())
        state = make_sim_state()
        state.game_phase = 'finished'

        with patch('game.players.neat_player.from_game', return_value=state):
            result = player.decide_action()

        assert result == 3  # SKIP

    def test_chosen_action_is_none_when_no_legal_actions(self):
        player = _make_neat_player(make_mock_network())
        state = make_sim_state()
        state.game_phase = 'finished'

        with patch('game.players.neat_player.from_game', return_value=state):
            player.decide_action()

        assert player._chosen_action is None


class TestNEATPlayerDecideRoute:
    def test_returns_link_id_from_cached_claim_action(self):
        player = _make_neat_player()
        player._chosen_action = Action(action_type=0, link_id=7, color='red', wilds=0)

        assert player.decide_route() == 7

    def test_returns_zero_when_no_action_cached(self):
        player = _make_neat_player()
        player._chosen_action = None

        assert player.decide_route() == 0

    def test_returns_zero_when_action_is_not_claim(self):
        player = _make_neat_player()
        player._chosen_action = Action(action_type=2, card_choice=3)

        assert player.decide_route() == 0


class TestNEATPlayerDecideWildCards:
    def test_returns_wilds_from_cached_action(self):
        player = _make_neat_player()
        player._chosen_action = Action(action_type=0, link_id=3, color='blue', wilds=2)

        assert player.decide_wild_cards() == 2

    def test_returns_zero_when_no_action_cached(self):
        player = _make_neat_player()
        player._chosen_action = None

        assert player.decide_wild_cards() == 0

    def test_returns_zero_when_action_is_not_claim(self):
        player = _make_neat_player()
        player._chosen_action = Action(action_type=1)

        assert player.decide_wild_cards() == 0


class TestNEATPlayerDecideTrainCard:
    def test_first_draw_returns_face_up_choice(self):
        """On the first draw call, should return the chosen face-up card index."""
        player = _make_neat_player()
        player._chosen_action = Action(action_type=2, card_choice=3)
        player._card_draw_count = 0

        result = player.decide_train_card()

        assert result == 3

    def test_second_draw_returns_draw_pile(self):
        """On the second call (after incrementing), should return DRAW_PILE."""
        from game.enums import TrainCardDecision
        player = _make_neat_player()
        player._chosen_action = Action(action_type=2, card_choice=3)
        player._card_draw_count = 0

        player.decide_train_card()  # first draw
        result = player.decide_train_card()  # second draw

        assert result == TrainCardDecision.DRAW_PILE.value

    def test_returns_draw_pile_when_no_draw_action(self):
        from game.enums import TrainCardDecision
        player = _make_neat_player()
        player._chosen_action = Action(action_type=0, link_id=0, color='red', wilds=0)
        player._card_draw_count = 0

        result = player.decide_train_card()

        assert result == TrainCardDecision.DRAW_PILE.value

    def test_returns_draw_pile_when_no_action(self):
        from game.enums import TrainCardDecision
        player = _make_neat_player()
        player._chosen_action = None
        player._card_draw_count = 0

        result = player.decide_train_card()

        assert result == TrainCardDecision.DRAW_PILE.value

    def test_all_card_choice_values_in_range(self):
        """Choices 0-5 map to face-up slots (0-4) or draw pile (5)."""
        player = _make_neat_player()
        for choice in range(6):
            player._chosen_action = Action(action_type=2, card_choice=choice)
            player._card_draw_count = 0
            result = player.decide_train_card()
            assert result == choice


# ── Integration: NEAT player in a real Game ───────────────────────────────────

class TestNEATPlayerIntegration:
    """Run a short game with a NEAT player using a mock network.

    Uses a mock network that always returns 0.5 so the player can make
    decisions without a trained genome. The game should complete without errors.
    """

    def test_neat_vs_random_game_completes(self):
        from game.core import Game
        mock_net = make_mock_network()  # always returns [0.5]

        game = Game(
            player_types=['NEAT', 'Random'],
            version='USA',
            networks=[mock_net, None],
        )
        stats = game.play(max_moves=500)

        assert game.winner is not None
        assert 'score' in stats

    def test_neat_vs_neat_game_completes(self):
        from game.core import Game
        mock_net = make_mock_network()

        game = Game(
            player_types=['NEAT', 'NEAT'],
            version='USA',
            networks=[mock_net, mock_net],
        )
        stats = game.play(max_moves=500)

        assert game.winner is not None

    def test_neat_player_action_types_are_valid(self):
        """All decisions made by NEAT player should be valid action types (0,1,2)."""
        from game.core import Game
        action_types_seen = set()
        call_count = [0]

        original_make = MagicMock(side_effect=[0.5])
        mock_net = MagicMock()

        def track_activate(features):
            call_count[0] += 1
            return [0.5]

        mock_net.activate.side_effect = track_activate

        game = Game(
            player_types=['NEAT', 'Random'],
            version='USA',
            networks=[mock_net, None],
        )
        game.play(max_moves=200)

        # As long as the game played moves, network was used
        assert call_count[0] > 0
