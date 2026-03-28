"""Tests for mcts/determinize.py and game/sim_game.py::clone().

Determinization invariants (from IS-MCTS correctness requirements):
- The perspective player's own hand is never modified.
- Each opponent's hand size is preserved.
- Total number of cards in the game is conserved.
- Opponent ticket counts are preserved.
- Route ownership is unchanged.
- Scores and train counts are unchanged.

Clone invariants:
- Mutations to the clone do not affect the original.
- Static data (route_info) is shared, not copied.
"""
import pytest

from game.sim_game import SimState, RouteInfo, clone
from game.ticket_deck import Ticket
from game.config import USAConfig
from mcts.determinize import determinize


# ── helpers ───────────────────────────────────────────────────────────────────

def make_state(num_players: int = 2) -> SimState:
    s = SimState()
    s.num_players = num_players
    s.config = USAConfig()
    s.route_values = dict(s.config.ROUTE_VALUES)
    s.wild_card_restriction = True
    s.min_train_figures = 3
    s.cards_per_draw = 2
    s.hands = [{} for _ in range(num_players)]
    s.tickets = [{} for _ in range(num_players)]
    s.scores = [0] * num_players
    s.trains = [45] * num_players
    s.current_player = 0
    s.game_phase = 'running'
    s.last_player = None
    return s


def total_cards(state: SimState) -> int:
    """Sum of all cards in the game (hands + draw + discard + face_up)."""
    total = 0
    for hand in state.hands:
        total += sum(hand.values())
    total += len(state.draw_pile)
    total += len(state.discard_pile)
    total += sum(1 for c in state.face_up_cards if c is not None)
    return total


# ── clone() ───────────────────────────────────────────────────────────────────

class TestClone:
    def test_clone_is_independent_from_original_hands(self):
        s = make_state()
        s.hands[0] = {'red': 3, 'blue': 2}
        c = clone(s)

        c.hands[0]['red'] = 99

        assert s.hands[0]['red'] == 3

    def test_clone_is_independent_from_original_draw_pile(self):
        s = make_state()
        s.draw_pile = ['red', 'blue', 'green']
        c = clone(s)

        c.draw_pile.append('extra')

        assert len(s.draw_pile) == 3

    def test_clone_is_independent_from_original_tickets(self):
        s = make_state()
        ticket = Ticket('A', 'B', 5)
        s.tickets[0] = {ticket: False}
        c = clone(s)

        c.tickets[0][ticket] = True

        assert s.tickets[0][ticket] is False

    def test_clone_is_independent_from_original_route_owners(self):
        s = make_state()
        s.route_owners[0] = None
        c = clone(s)

        c.route_owners[0] = 1

        assert s.route_owners[0] is None

    def test_clone_is_independent_from_original_scores(self):
        s = make_state()
        s.scores = [10, 20]
        c = clone(s)

        c.scores[0] = 999

        assert s.scores[0] == 10

    def test_clone_is_independent_from_original_trains(self):
        s = make_state()
        s.trains = [45, 45]
        c = clone(s)

        c.trains[1] = 0

        assert s.trains[1] == 45

    def test_clone_shares_route_info(self):
        """route_info is frozen and shared to avoid copying large static data."""
        s = make_state()
        ri = RouteInfo(0, 0, 'A', 'B', 3, 'red')
        s.route_info = [ri]
        c = clone(s)

        assert c.route_info is s.route_info

    def test_clone_copies_scalar_fields(self):
        s = make_state()
        s.current_player = 1
        s.game_phase = 'last_round'
        s.last_player = 0
        c = clone(s)

        assert c.current_player == 1
        assert c.game_phase == 'last_round'
        assert c.last_player == 0

    def test_clone_face_up_cards_independent(self):
        s = make_state()
        s.face_up_cards = ['red', 'blue', None, None, None]
        c = clone(s)

        c.face_up_cards[0] = 'wild'

        assert s.face_up_cards[0] == 'red'

    def test_modifying_clone_does_not_corrupt_mcts_parent(self):
        """Simulates the MCTS use-case: clone a state, apply actions to clone,
        verify that the original is untouched."""
        s = make_state()
        s.hands[0] = {'red': 5}
        s.draw_pile = ['blue', 'green']
        s.scores = [7, 3]
        s.trains = [40, 45]

        c = clone(s)
        c.hands[0]['red'] = 0
        c.draw_pile.clear()
        c.scores[0] = 100
        c.trains[0] = 0

        assert s.hands[0]['red'] == 5
        assert len(s.draw_pile) == 2
        assert s.scores[0] == 7
        assert s.trains[0] == 40


# ── determinize(): own hand unchanged ────────────────────────────────────────

class TestDeterminizeOwnHand:
    """The perspective player knows their own hand exactly — it must not change."""

    def test_own_hand_unchanged_2_player(self):
        s = make_state(num_players=2)
        s.hands[0] = {'red': 3, 'blue': 2, 'wild': 1}
        s.hands[1] = {'green': 4}
        s.draw_pile = ['yellow', 'black', 'orange']
        s.face_up_cards = ['pink', None, None, None, None]

        det = determinize(s, perspective=0)

        assert det.hands[0] == s.hands[0]

    def test_own_hand_unchanged_3_player(self):
        s = make_state(num_players=3)
        s.hands[0] = {'blue': 5, 'wild': 2}
        s.hands[1] = {'red': 3}
        s.hands[2] = {'green': 2}
        s.draw_pile = ['yellow', 'black']

        det = determinize(s, perspective=0)

        assert det.hands[0] == s.hands[0]

    def test_determinize_does_not_mutate_original(self):
        s = make_state(num_players=2)
        s.hands[0] = {'red': 3}
        s.hands[1] = {'blue': 2}
        s.draw_pile = ['green', 'yellow']
        original_p0 = dict(s.hands[0])
        original_p1_size = sum(s.hands[1].values())

        determinize(s, perspective=0)

        assert s.hands[0] == original_p0
        assert sum(s.hands[1].values()) == original_p1_size


# ── determinize(): opponent hand sizes preserved ──────────────────────────────

class TestDeterminizeOpponentHandSizes:
    """We know how many cards each opponent holds; determinization must
    redistribute cards while keeping those counts."""

    def test_opponent_hand_size_preserved_2_player(self):
        s = make_state(num_players=2)
        s.hands[0] = {'red': 2}
        s.hands[1] = {'blue': 3, 'green': 2}  # total 5
        s.draw_pile = ['yellow', 'black', 'orange']

        det = determinize(s, perspective=0)

        assert sum(det.hands[1].values()) == 5

    def test_opponent_hand_size_preserved_3_player(self):
        s = make_state(num_players=3)
        s.hands[0] = {'red': 1}
        s.hands[1] = {'blue': 4}       # total 4
        s.hands[2] = {'green': 2, 'wild': 1}  # total 3
        s.draw_pile = ['yellow', 'black', 'orange', 'pink']

        det = determinize(s, perspective=0)

        assert sum(det.hands[1].values()) == 4
        assert sum(det.hands[2].values()) == 3


# ── determinize(): total card conservation ────────────────────────────────────

class TestDeterminizeCardConservation:
    """No card should be created or destroyed by determinization."""

    def test_total_cards_conserved_2_player(self):
        s = make_state(num_players=2)
        s.hands[0] = {'red': 2, 'wild': 1}
        s.hands[1] = {'blue': 3}
        s.draw_pile = ['green', 'yellow', 'black']
        s.discard_pile = ['orange']
        s.face_up_cards = ['pink', None, None, None, None]
        n = total_cards(s)

        det = determinize(s, perspective=0)

        assert total_cards(det) == n

    def test_total_cards_conserved_3_player(self):
        s = make_state(num_players=3)
        s.hands[0] = {'red': 3}
        s.hands[1] = {'blue': 2}
        s.hands[2] = {'green': 4}
        s.draw_pile = ['yellow', 'black', 'orange', 'pink']
        s.discard_pile = []
        s.face_up_cards = ['white', 'red', None, None, None]
        n = total_cards(s)

        det = determinize(s, perspective=0)

        assert total_cards(det) == n


# ── determinize(): route ownership and scores unchanged ───────────────────────

class TestDeterminizeStaticFields:
    def test_route_ownership_unchanged(self):
        s = make_state(num_players=2)
        s.route_owners[5] = 0
        s.route_owners[7] = 1
        s.hands[1] = {'red': 2}
        s.draw_pile = ['blue']

        det = determinize(s, perspective=0)

        assert det.route_owners[5] == 0
        assert det.route_owners[7] == 1

    def test_scores_unchanged(self):
        s = make_state(num_players=2)
        s.scores = [15, 8]
        s.hands[1] = {'red': 1}
        s.draw_pile = ['blue']

        det = determinize(s, perspective=0)

        assert det.scores == [15, 8]

    def test_trains_unchanged(self):
        s = make_state(num_players=2)
        s.trains = [40, 38]
        s.hands[1] = {'red': 1}
        s.draw_pile = ['blue']

        det = determinize(s, perspective=0)

        assert det.trains == [40, 38]

    def test_game_phase_unchanged(self):
        s = make_state(num_players=2)
        s.game_phase = 'last_round'
        s.last_player = 1
        s.hands[1] = {'red': 2}
        s.draw_pile = ['blue']

        det = determinize(s, perspective=0)

        assert det.game_phase == 'last_round'
        assert det.last_player == 1

    def test_current_player_unchanged(self):
        s = make_state(num_players=2)
        s.current_player = 1
        s.hands[0] = {'blue': 2}
        s.draw_pile = ['red']

        det = determinize(s, perspective=1)

        assert det.current_player == 1


# ── determinize(): ticket count preservation ─────────────────────────────────

class TestDeterminizeTickets:
    """We know how many tickets each opponent holds; counts must be preserved."""

    def test_opponent_ticket_count_preserved(self):
        s = make_state(num_players=2)
        t1, t2, t3 = Ticket('A', 'B', 5), Ticket('C', 'D', 7), Ticket('E', 'F', 9)
        s.tickets[0] = {}
        s.tickets[1] = {t1: False, t2: True}  # opponent has 2 tickets
        s.ticket_deck_remaining = [t3]
        s.hands[1] = {'red': 1}
        s.draw_pile = ['blue']

        det = determinize(s, perspective=0)

        assert len(det.tickets[1]) == 2

    def test_own_ticket_unchanged(self):
        s = make_state(num_players=2)
        t1, t2 = Ticket('A', 'B', 5), Ticket('C', 'D', 7)
        s.tickets[0] = {t1: True}   # perspective player's ticket
        s.tickets[1] = {t2: False}
        s.ticket_deck_remaining = [Ticket('E', 'F', 9)]
        s.hands[1] = {'red': 1}
        s.draw_pile = ['blue']

        det = determinize(s, perspective=0)

        assert t1 in det.tickets[0]
        assert det.tickets[0][t1] is True

    def test_determinize_multiple_times_gives_different_results(self):
        """Randomization should produce variation across calls."""
        s = make_state(num_players=2)
        s.hands[0] = {'red': 1}
        s.hands[1] = {'blue': 2, 'green': 3}
        s.draw_pile = ['yellow', 'black', 'orange', 'pink', 'white']

        results = set()
        for _ in range(20):
            det = determinize(s, perspective=0)
            # Represent opponent hand as a frozenset of (color, count) pairs
            hand_repr = frozenset(det.hands[1].items())
            results.add(hand_repr)

        # With enough cards and randomization, we should see > 1 different result
        assert len(results) > 1
