"""Tests for the Ticket to Ride simulation engine (sim_game.py).

Rules source: docs/7f-ticket-to-ride-rulebook.pdf

Covers:
- Route scoring table (official values)
- Claiming routes (card consumption, train decrement, ownership, discard)
- Grey routes (any color allowed)
- Double routes (closed in 2-3 player games)
- Drawing train cards (2 per turn; face-up wild = 1 card only)
- Blind-drawn wild still allows second draw
- Wild threshold (≥3 face-up wilds → discard all and replace)
- Discard pile reshuffled when draw pile exhausted
- Drawing destination tickets
- Last round trigger (0, 1, or 2 trains left)
- Last round: every player including triggering player gets one final turn
- Terminal state and no legal actions after game ends
- Ticket completion via BFS path check
- Score evaluation (completed tickets add, incomplete subtract)
- Longest continuous path bonus (+10, ties share it)
- Turn advancement and player wrapping
"""
import pytest

from game.sim_game import (
    SimState, Action, RouteInfo,
    apply_action, get_legal_actions, is_terminal, evaluate,
    _advance_player, _draw_from_pile, _fill_face_up,
    _is_ticket_completed, _calculate_longest_path,
)
from game.ticket_deck import Ticket
from game.config import USAConfig


# ── Fixtures / helpers ────────────────────────────────────────────────────────

def make_state(num_players: int = 2) -> SimState:
    """Return a minimal two-player SimState with an empty board and no cards."""
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


def add_route(state: SimState, link_id: int, route_id: int,
              city_from: str, city_to: str, weight: int,
              edge_color: str = 'grey') -> RouteInfo:
    """Register a route on the board (unclaimed)."""
    ri = RouteInfo(
        link_id=link_id, route_id=route_id,
        city_from=city_from, city_to=city_to,
        weight=weight, edge_color=edge_color,
    )
    state.route_info.append(ri)
    state.route_owners[link_id] = None
    state.route_ids_by_route_id.setdefault(route_id, []).append(link_id)
    return ri


# ── Route Scoring Table ───────────────────────────────────────────────────────

class TestRouteScoringTable:
    """Rule: 1→1, 2→2, 3→4, 4→7, 5→10, 6→15 points."""

    @pytest.mark.parametrize("length,points", [
        (1, 1), (2, 2), (3, 4), (4, 7), (5, 10), (6, 15),
    ])
    def test_config_matches_rulebook(self, length, points):
        assert USAConfig().ROUTE_VALUES[length] == points

    @pytest.mark.parametrize("length,points", [
        (1, 1), (2, 2), (3, 4), (4, 7), (5, 10), (6, 15),
    ])
    def test_claiming_route_awards_correct_points(self, length, points):
        s = make_state()
        add_route(s, 0, 0, 'A', 'B', length, 'blue')
        s.hands[0]['blue'] = length

        apply_action(s, Action(action_type=0, link_id=0, color='blue', wilds=0))

        assert s.scores[0] == points


# ── Claiming Routes ───────────────────────────────────────────────────────────

class TestClaimRoute:
    """Rules: play matching cards, place trains, record score, discard cards."""

    def test_cards_consumed_from_hand(self):
        s = make_state()
        add_route(s, 0, 0, 'A', 'B', 3, 'blue')
        s.hands[0]['blue'] = 5

        apply_action(s, Action(action_type=0, link_id=0, color='blue', wilds=0))

        assert s.hands[0].get('blue', 0) == 2

    def test_wild_cards_consumed_alongside_color_cards(self):
        s = make_state()
        add_route(s, 0, 0, 'A', 'B', 4, 'red')
        s.hands[0] = {'red': 2, 'wild': 3}

        apply_action(s, Action(action_type=0, link_id=0, color='red', wilds=2))

        assert s.hands[0].get('red', 0) == 0
        assert s.hands[0].get('wild', 0) == 1

    def test_trains_decremented_by_route_length(self):
        s = make_state()
        add_route(s, 0, 0, 'A', 'B', 4, 'green')
        s.hands[0]['green'] = 4

        apply_action(s, Action(action_type=0, link_id=0, color='green', wilds=0))

        assert s.trains[0] == 41  # 45 - 4

    def test_route_becomes_owned_by_player(self):
        s = make_state()
        add_route(s, 0, 0, 'A', 'B', 2, 'yellow')
        s.hands[0]['yellow'] = 2

        apply_action(s, Action(action_type=0, link_id=0, color='yellow', wilds=0))

        assert s.route_owners[0] == 0

    def test_used_cards_go_to_discard_pile(self):
        s = make_state()
        add_route(s, 0, 0, 'A', 'B', 2, 'blue')
        s.hands[0]['blue'] = 2

        apply_action(s, Action(action_type=0, link_id=0, color='blue', wilds=0))

        assert s.discard_pile.count('blue') == 2

    def test_wild_cards_go_to_discard_pile(self):
        s = make_state()
        add_route(s, 0, 0, 'A', 'B', 3, 'red')
        s.hands[0] = {'red': 2, 'wild': 1}

        apply_action(s, Action(action_type=0, link_id=0, color='red', wilds=1))

        assert 'wild' in s.discard_pile

    def test_owned_route_not_in_legal_actions(self):
        s = make_state()
        add_route(s, 0, 0, 'A', 'B', 1, 'black')
        s.hands[0] = {'black': 1}
        s.hands[1] = {'black': 1}

        apply_action(s, Action(action_type=0, link_id=0, color='black', wilds=0))
        # Now player 1's turn — link 0 is owned, must not appear
        legal = get_legal_actions(s)
        assert not any(a.action_type == 0 and a.link_id == 0 for a in legal)

    def test_cannot_claim_without_enough_cards(self):
        s = make_state()
        add_route(s, 0, 0, 'A', 'B', 3, 'red')
        s.hands[0]['red'] = 1  # need 3

        legal = get_legal_actions(s)
        assert not any(a.action_type == 0 for a in legal)

    def test_cannot_claim_without_enough_trains(self):
        s = make_state()
        add_route(s, 0, 0, 'A', 'B', 4, 'blue')
        s.hands[0]['blue'] = 4
        s.trains[0] = 3  # route needs 4

        legal = get_legal_actions(s)
        assert not any(a.action_type == 0 for a in legal)

    def test_grey_route_claimable_with_any_single_color(self):
        s = make_state()
        add_route(s, 0, 0, 'A', 'B', 2, 'grey')
        s.hands[0]['orange'] = 2

        legal = get_legal_actions(s)
        claim = [a for a in legal if a.action_type == 0 and a.link_id == 0]

        assert len(claim) == 1
        assert claim[0].color == 'orange'

    def test_grey_route_picks_option_with_fewest_wilds(self):
        """Grey route action uses the color that minimises wild card usage."""
        s = make_state()
        add_route(s, 0, 0, 'A', 'B', 3, 'grey')
        # blue: needs 1 wild; red: needs 0 wilds → red is better
        s.hands[0] = {'red': 3, 'blue': 2, 'wild': 2}

        legal = get_legal_actions(s)
        claim = [a for a in legal if a.action_type == 0 and a.link_id == 0]

        assert len(claim) == 1
        assert claim[0].color == 'red'
        assert claim[0].wilds == 0


# ── Double Routes ─────────────────────────────────────────────────────────────

class TestDoubleRoutes:
    """Rule: In 2 or 3 player games, only one of the double routes can be used."""

    def _setup_double_route(self, num_players: int) -> SimState:
        s = make_state(num_players=num_players)
        add_route(s, link_id=0, route_id=42, city_from='A', city_to='B',
                  weight=2, edge_color='blue')
        add_route(s, link_id=1, route_id=42, city_from='A', city_to='B',
                  weight=2, edge_color='red')
        s.hands = [{'blue': 2, 'red': 2}] + [{'red': 4}] * (num_players - 1)
        return s

    def test_second_link_blocked_in_2_player_game(self):
        s = self._setup_double_route(2)

        apply_action(s, Action(action_type=0, link_id=0, color='blue', wilds=0))
        # Player 1's turn
        legal = get_legal_actions(s)
        assert not any(a.action_type == 0 and a.link_id == 1 for a in legal)

    def test_second_link_blocked_in_3_player_game(self):
        s = self._setup_double_route(3)

        apply_action(s, Action(action_type=0, link_id=0, color='blue', wilds=0))
        # Player 1's turn
        legal = get_legal_actions(s)
        assert not any(a.action_type == 0 and a.link_id == 1 for a in legal)

    def test_player_cannot_claim_both_links(self):
        s = make_state(num_players=2)
        add_route(s, link_id=0, route_id=7, city_from='A', city_to='B',
                  weight=1, edge_color='blue')
        add_route(s, link_id=1, route_id=7, city_from='A', city_to='B',
                  weight=1, edge_color='red')
        s.hands = [{'blue': 1, 'red': 2}, {}]
        s.draw_pile = ['green']

        apply_action(s, Action(action_type=0, link_id=0, color='blue', wilds=0))
        # Player 1 draws to advance back to player 0
        apply_action(s, Action(action_type=2, card_choice=5))
        # Player 0 — link 1 must be unavailable (they own the sibling)
        legal = get_legal_actions(s)
        assert not any(a.action_type == 0 and a.link_id == 1 for a in legal)


# ── Drawing Train Cards ───────────────────────────────────────────────────────

class TestDrawTrainCards:
    """Rules: draw 2 cards per turn; taking a face-up Locomotive ends the turn
    (counts as a full draw of only 1 card)."""

    def test_draw_from_pile_gives_two_cards(self):
        s = make_state()
        s.draw_pile = ['red', 'blue', 'green']

        apply_action(s, Action(action_type=2, card_choice=5))

        assert sum(s.hands[0].values()) == 2

    def test_draw_face_up_non_wild_gives_two_cards(self):
        s = make_state()
        # Full face-up display so _fill_face_up only needs to replace 1 card,
        # leaving a second card in draw_pile for the mandatory second draw.
        s.face_up_cards = ['red', 'blue', 'green', 'yellow', 'black']
        s.draw_pile = ['orange', 'pink']  # first used for refill, second for 2nd draw

        apply_action(s, Action(action_type=2, card_choice=0))

        assert sum(s.hands[0].values()) == 2

    def test_draw_face_up_wild_gives_only_one_card(self):
        """Rule: If a Locomotive is one of the five face-up cards,
        the player who draws it may only draw one card, instead of two."""
        s = make_state()
        s.face_up_cards = ['wild', None, None, None, None]
        s.draw_pile = ['blue', 'green', 'red']

        apply_action(s, Action(action_type=2, card_choice=0))

        assert sum(s.hands[0].values()) == 1
        assert s.hands[0].get('wild', 0) == 1

    def test_blind_draw_wild_still_counts_as_one_of_two(self):
        """Rule: If a Locomotive is drawn blind from the deck it still counts
        as a single card and the player may still draw two cards total."""
        s = make_state()
        s.draw_pile = ['wild', 'red']  # wild drawn first (top of stack)

        apply_action(s, Action(action_type=2, card_choice=5))

        assert sum(s.hands[0].values()) == 2

    def test_face_up_card_replaced_after_draw(self):
        s = make_state()
        s.face_up_cards = ['red', 'blue', 'green', 'yellow', 'black']
        s.draw_pile = ['orange']

        apply_action(s, Action(action_type=2, card_choice=0))  # take 'red'

        assert s.face_up_cards[0] == 'orange'

    def test_discard_reshuffled_when_draw_pile_empty(self):
        """Rule: When the deck is exhausted, discards are reshuffled into a
        new draw pile deck."""
        s = make_state()
        s.draw_pile = []
        s.discard_pile = ['red', 'blue', 'green', 'yellow']

        card = _draw_from_pile(s)

        assert card is not None
        assert len(s.discard_pile) == 0  # discard consumed into draw pile

    def test_draw_unavailable_when_both_piles_empty(self):
        s = make_state()
        s.draw_pile = []
        s.discard_pile = []
        s.face_up_cards = [None] * 5

        legal = get_legal_actions(s)
        assert not any(a.action_type == 2 for a in legal)


# ── Wild Card Face-Up Threshold ───────────────────────────────────────────────

class TestWildCardThreshold:
    """Rule: If at any time three of the five face-up cards are Locomotives,
    all five cards are immediately discarded and five new ones are turned face-up."""

    def test_three_wilds_face_up_triggers_redeal(self):
        s = make_state()
        s.draw_pile = ['red', 'blue', 'green', 'yellow', 'black', 'orange', 'pink']
        s.face_up_cards = ['wild', 'wild', 'wild', None, None]
        s.discard_pile = []

        _fill_face_up(s)

        wild_count = sum(1 for c in s.face_up_cards if c == 'wild')
        assert wild_count < 3

    def test_discarded_wilds_go_to_discard_pile(self):
        s = make_state()
        # Provide enough non-wild cards so the draw pile never runs out and the
        # discard pile is never reshuffled back — wilds remain in discard_pile.
        s.draw_pile = ['red', 'blue', 'green', 'yellow', 'black', 'orange', 'pink', 'white']
        s.face_up_cards = ['wild', 'wild', 'wild', None, None]
        s.discard_pile = []

        _fill_face_up(s)

        assert 'wild' in s.discard_pile

    def test_two_wilds_face_up_does_not_trigger_redeal(self):
        s = make_state()
        s.draw_pile = ['red', 'blue']
        s.face_up_cards = ['wild', 'wild', 'blue', None, None]
        s.discard_pile = []

        _fill_face_up(s)

        wild_count = sum(1 for c in s.face_up_cards if c == 'wild')
        assert wild_count == 2


# ── Drawing Destination Tickets ───────────────────────────────────────────────

class TestDrawTickets:
    """Rules: draw 3 tickets, keep at least 1; returned cards go to the bottom.
    Simulation simplification: all drawn tickets are kept."""

    def test_draw_takes_up_to_three_tickets_from_top(self):
        s = make_state()
        t1, t2, t3 = Ticket('A', 'B', 5), Ticket('C', 'D', 7), Ticket('E', 'F', 10)
        s.ticket_deck_remaining = [t1, t2, t3, Ticket('G', 'H', 4)]

        apply_action(s, Action(action_type=1))

        assert t1 in s.tickets[0]
        assert t2 in s.tickets[0]
        assert t3 in s.tickets[0]

    def test_draw_removes_tickets_from_deck(self):
        s = make_state()
        s.ticket_deck_remaining = [Ticket('A', 'B', i) for i in range(5)]

        apply_action(s, Action(action_type=1))

        assert len(s.ticket_deck_remaining) == 2  # 5 - 3

    def test_draw_fewer_than_three_when_deck_small(self):
        s = make_state()
        t1 = Ticket('A', 'B', 5)
        s.ticket_deck_remaining = [t1]

        apply_action(s, Action(action_type=1))

        assert t1 in s.tickets[0]
        assert len(s.ticket_deck_remaining) == 0

    def test_draw_tickets_not_available_when_deck_empty(self):
        s = make_state()
        s.ticket_deck_remaining = []

        legal = get_legal_actions(s)
        assert not any(a.action_type == 1 for a in legal)

    def test_draw_tickets_available_when_deck_nonempty(self):
        s = make_state()
        s.ticket_deck_remaining = [Ticket('A', 'B', 5)]

        legal = get_legal_actions(s)
        assert any(a.action_type == 1 for a in legal)


# ── Last Round ────────────────────────────────────────────────────────────────

class TestLastRound:
    """Rule: When a player's stock gets down to 0, 1, or 2 trains at the end
    of their turn, each player — including that player — gets one final turn."""

    @pytest.mark.parametrize("trains_left", [0, 1, 2])
    def test_last_round_triggered_at_threshold(self, trains_left):
        s = make_state()
        s.trains[0] = trains_left

        _advance_player(s, acted_pid=0)

        assert s.game_phase == 'last_round'
        assert s.last_player == 0

    def test_last_round_not_triggered_above_threshold(self):
        s = make_state()
        s.trains[0] = 4  # above min_train_figures=3

        _advance_player(s, acted_pid=0)

        assert s.game_phase == 'running'

    def test_game_not_finished_before_last_player_final_turn(self):
        """The triggering player must not be skipped — game ends only after
        they complete their own final turn."""
        s = make_state(num_players=2)
        s.trains[0] = 2

        _advance_player(s, acted_pid=0)  # triggers last_round; advance to p1
        assert s.game_phase == 'last_round'

        _advance_player(s, acted_pid=1)  # p1 final turn; advance to p0
        assert s.game_phase != 'finished', (
            "Game must not end before last_player (p0) takes their final turn"
        )

    def test_game_finishes_after_last_player_final_turn(self):
        s = make_state(num_players=2)
        s.trains[0] = 2

        _advance_player(s, acted_pid=0)  # triggers, advance to p1
        _advance_player(s, acted_pid=1)  # p1 final turn, advance to p0
        _advance_player(s, acted_pid=0)  # p0 final turn → finished

        assert s.game_phase == 'finished'

    def test_three_player_all_get_final_turn(self):
        """All three players, including the triggering player, get a final turn."""
        s = make_state(num_players=3)
        s.trains[1] = 1  # player 1 triggers

        # Simulate player 0's turn (no trigger)
        _advance_player(s, acted_pid=0)
        # Player 1 triggers last_round
        _advance_player(s, acted_pid=1)
        assert s.game_phase == 'last_round'
        assert s.last_player == 1

        # Player 2 final turn
        _advance_player(s, acted_pid=2)
        assert s.game_phase != 'finished'

        # Player 0 final turn
        _advance_player(s, acted_pid=0)
        assert s.game_phase != 'finished'

        # Player 1 (last_player) final turn → finished
        _advance_player(s, acted_pid=1)
        assert s.game_phase == 'finished'


# ── Terminal State ────────────────────────────────────────────────────────────

class TestTerminalState:
    def test_not_terminal_when_running(self):
        assert not is_terminal(make_state())

    def test_not_terminal_in_last_round(self):
        s = make_state()
        s.game_phase = 'last_round'
        assert not is_terminal(s)

    def test_terminal_when_finished(self):
        s = make_state()
        s.game_phase = 'finished'
        assert is_terminal(s)

    def test_no_legal_actions_when_terminal(self):
        s = make_state()
        s.game_phase = 'finished'
        s.draw_pile = ['red', 'blue']
        s.ticket_deck_remaining = [Ticket('A', 'B', 5)]
        add_route(s, 0, 0, 'A', 'B', 1, 'blue')
        s.hands[0]['blue'] = 1

        assert get_legal_actions(s) == []


# ── Ticket Completion ─────────────────────────────────────────────────────────

class TestTicketCompletion:
    """Rule: completing a continuous path between ticket cities scores the
    ticket's point value; failing to do so deducts those points."""

    def test_direct_route_completes_ticket(self):
        s = make_state()
        add_route(s, 0, 0, 'Chicago', 'Dallas', 7, 'grey')
        s.route_owners[0] = 0
        ticket = Ticket('Chicago', 'Dallas', 7)

        assert _is_ticket_completed(s, 0, ticket)

    def test_indirect_path_completes_ticket(self):
        s = make_state()
        add_route(s, 0, 0, 'A', 'B', 2, 'grey')
        add_route(s, 1, 1, 'B', 'C', 3, 'grey')
        s.route_owners[0] = s.route_owners[1] = 0
        ticket = Ticket('A', 'C', 5)

        assert _is_ticket_completed(s, 0, ticket)

    def test_broken_path_does_not_complete_ticket(self):
        s = make_state()
        add_route(s, 0, 0, 'A', 'B', 2, 'grey')
        add_route(s, 1, 1, 'B', 'C', 3, 'grey')
        s.route_owners[0] = 0
        s.route_owners[1] = 1  # owned by opponent — path broken
        ticket = Ticket('A', 'C', 5)

        assert not _is_ticket_completed(s, 0, ticket)

    def test_no_routes_does_not_complete_ticket(self):
        s = make_state()
        assert not _is_ticket_completed(s, 0, Ticket('A', 'B', 5))

    def test_evaluate_adds_completed_ticket_points(self):
        s = make_state()
        s.tickets[0] = {Ticket('A', 'B', 8): True}
        s.game_phase = 'finished'

        assert evaluate(s)[0] == 8

    def test_evaluate_subtracts_incomplete_ticket_points(self):
        s = make_state()
        s.tickets[0] = {Ticket('A', 'B', 8): False}
        s.game_phase = 'finished'

        assert evaluate(s)[0] == -8

    def test_evaluate_mixed_tickets(self):
        s = make_state()
        s.tickets[0] = {Ticket('A', 'B', 10): True, Ticket('C', 'D', 6): False}
        s.game_phase = 'finished'

        assert evaluate(s)[0] == 4  # +10 − 6

    def test_evaluate_includes_route_score_already_accumulated(self):
        s = make_state()
        s.scores[0] = 7  # route points accumulated during play
        s.tickets[0] = {Ticket('A', 'B', 5): True}
        s.game_phase = 'finished'

        assert evaluate(s)[0] == 12  # 7 route + 5 ticket


# ── Longest Continuous Path Bonus ─────────────────────────────────────────────

class TestLongestPath:
    """Rule: player with the longest continuous path receives a +10 bonus.
    In case of a tie, all tied players score the bonus."""

    def test_bonus_awarded_to_player_with_longer_path(self):
        s = make_state()
        # p0: A-B(3) + B-C(3) = path of 6
        add_route(s, 0, 0, 'A', 'B', 3, 'grey')
        add_route(s, 1, 1, 'B', 'C', 3, 'grey')
        # p1: X-Y(2) = path of 2
        add_route(s, 2, 2, 'X', 'Y', 2, 'grey')
        s.route_owners[0] = s.route_owners[1] = 0
        s.route_owners[2] = 1
        s.game_phase = 'finished'

        scores = evaluate(s)

        assert scores[0] == 10   # bonus
        assert scores[1] == 0    # no bonus

    def test_bonus_shared_on_tie(self):
        """Rule: In the case of a tie, all tied players score the 10 point bonus."""
        s = make_state()
        add_route(s, 0, 0, 'A', 'B', 3, 'grey')
        add_route(s, 1, 1, 'C', 'D', 3, 'grey')
        s.route_owners[0] = 0
        s.route_owners[1] = 1
        s.game_phase = 'finished'

        scores = evaluate(s)

        assert scores[0] == 10
        assert scores[1] == 10

    def test_no_bonus_with_no_routes(self):
        s = make_state()
        s.game_phase = 'finished'

        scores = evaluate(s)

        assert scores[0] == 0
        assert scores[1] == 0

    def test_calculate_single_route_path(self):
        s = make_state()
        add_route(s, 0, 0, 'A', 'B', 5, 'grey')
        s.route_owners[0] = 0

        assert _calculate_longest_path(s, 0) == 5

    def test_calculate_path_picks_longer_branch(self):
        """A-B(3), B-C(2), B-D(4) → longest path is A-B-D = 7."""
        s = make_state()
        add_route(s, 0, 0, 'A', 'B', 3, 'grey')
        add_route(s, 1, 1, 'B', 'C', 2, 'grey')
        add_route(s, 2, 2, 'B', 'D', 4, 'grey')
        s.route_owners[0] = s.route_owners[1] = s.route_owners[2] = 0

        assert _calculate_longest_path(s, 0) == 7

    def test_calculate_path_ignores_opponent_routes(self):
        s = make_state()
        add_route(s, 0, 0, 'A', 'B', 3, 'grey')
        add_route(s, 1, 1, 'B', 'C', 4, 'grey')
        s.route_owners[0] = 0
        s.route_owners[1] = 1  # opponent owns this

        assert _calculate_longest_path(s, 0) == 3

    def test_calculate_no_routes_returns_zero(self):
        assert _calculate_longest_path(make_state(), 0) == 0


# ── Legal Actions ─────────────────────────────────────────────────────────────

class TestLegalActions:
    def test_draw_pile_action_available_with_cards(self):
        s = make_state()
        s.draw_pile = ['red']

        legal = get_legal_actions(s)
        assert any(a.action_type == 2 and a.card_choice == 5 for a in legal)

    def test_draw_pile_action_unavailable_when_empty(self):
        s = make_state()
        s.draw_pile = []
        s.discard_pile = []
        s.face_up_cards = [None] * 5

        legal = get_legal_actions(s)
        assert not any(a.action_type == 2 for a in legal)

    def test_no_duplicate_face_up_actions_for_same_color(self):
        """Two face-up cards of the same color should produce only one draw action."""
        s = make_state()
        s.face_up_cards = ['red', 'red', 'blue', None, None]
        s.draw_pile = []
        s.discard_pile = []

        legal = get_legal_actions(s)
        face_up_draws = [a for a in legal if a.action_type == 2 and a.card_choice != 5]
        colors_offered = [s.face_up_cards[a.card_choice] for a in face_up_draws]

        assert colors_offered.count('red') == 1

    def test_face_up_wild_produces_separate_draw_action(self):
        s = make_state()
        s.face_up_cards = ['wild', 'red', None, None, None]
        s.draw_pile = []
        s.discard_pile = []

        legal = get_legal_actions(s)
        wild_actions = [a for a in legal if a.action_type == 2
                        and a.card_choice != 5
                        and s.face_up_cards[a.card_choice] == 'wild']

        assert len(wild_actions) == 1


# ── Turn Advancement ──────────────────────────────────────────────────────────

class TestTurnAdvancement:
    def test_turn_advances_to_next_player(self):
        s = make_state(num_players=3)
        s.current_player = 0
        _advance_player(s, acted_pid=0)
        assert s.current_player == 1

    def test_turn_wraps_around_at_last_player(self):
        s = make_state(num_players=3)
        s.current_player = 2
        _advance_player(s, acted_pid=2)
        assert s.current_player == 0

    def test_current_player_changes_after_action(self):
        s = make_state(num_players=2)
        s.draw_pile = ['red', 'blue', 'green']
        assert s.current_player == 0

        apply_action(s, Action(action_type=2, card_choice=5))

        assert s.current_player == 1
