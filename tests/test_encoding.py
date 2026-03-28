"""Tests for alphazero/encoding.py.

Verifies that the state/action encoding used as neural network input is
correct: right shape, values in range, correct perspective, invertible
action mapping, and valid action masking.
"""
import pytest
import torch

from alphazero.encoding import (
    encode_state, action_to_index, index_to_action, get_action_mask,
    STATE_SIZE, ACTION_SPACE_SIZE,
    CLAIM_ROUTE_OFFSET, DRAW_CARD_OFFSET, DRAW_TICKETS_OFFSET,
    NUM_LINKS, NUM_COLORS, COLOR_TO_IDX, CANONICAL_TICKETS,
)
from game.sim_game import SimState, Action, RouteInfo
from game.ticket_deck import Ticket
from game.config import USAConfig


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
    # Populate route_owners for all 100 links (all unclaimed)
    for i in range(NUM_LINKS):
        s.route_owners[i] = None
    return s


# ── State encoding: shape and range ──────────────────────────────────────────

class TestEncodeStateShape:
    def test_returns_tensor_of_correct_size(self):
        s = make_state()
        tensor = encode_state(s, perspective=0)
        assert tensor.shape == (STATE_SIZE,)

    def test_state_size_constant_is_389(self):
        assert STATE_SIZE == 389

    def test_all_values_in_unit_interval(self):
        """Every feature must lie in [0, 1] after the /110 normalisation fix."""
        s = make_state()
        # Give player 1 a large hand to exercise the normalisation
        s.hands[1] = {c: 10 for c in ['red', 'blue', 'green', 'yellow']}
        tensor = encode_state(s, perspective=0)
        assert float(tensor.min()) >= 0.0
        assert float(tensor.max()) <= 1.0

    def test_dtype_is_float32(self):
        s = make_state()
        tensor = encode_state(s, perspective=0)
        assert tensor.dtype == torch.float32


# ── State encoding: perspective correctness ───────────────────────────────────

class TestEncodeStatePerspective:
    """The encoding must correctly distinguish 'my' information from
    the opponent's partial information."""

    def test_own_hand_encoded_in_first_9_features(self):
        """Features 0-8 represent the perspective player's card counts."""
        s = make_state()
        s.hands[0] = {'red': 3}  # player 0 has 3 red cards

        tensor = encode_state(s, perspective=0)

        red_idx = COLOR_TO_IDX['red']
        assert tensor[red_idx].item() == pytest.approx(3 / 12.0)

    def test_opponent_hand_encoded_as_size_only(self):
        """Feature 9 is the total opponent hand size (not individual counts)."""
        s = make_state()
        s.hands[1] = {'red': 4, 'blue': 6}  # opponent has 10 cards total

        tensor = encode_state(s, perspective=0)

        opp_hand_feature = tensor[9].item()  # index 9 = after 9 own-hand features
        assert opp_hand_feature == pytest.approx(10 / 110.0)

    def test_perspective_swap_changes_own_hand_features(self):
        """Encoding from player 1's perspective should show player 1's hand
        in the own-hand slots (features 0-8)."""
        s = make_state()
        s.hands[0] = {'blue': 5}
        s.hands[1] = {'red': 3}

        t0 = encode_state(s, perspective=0)
        t1 = encode_state(s, perspective=1)

        blue_idx = COLOR_TO_IDX['blue']
        red_idx = COLOR_TO_IDX['red']
        # From p0's view: own hand has blue
        assert t0[blue_idx].item() == pytest.approx(5 / 12.0)
        assert t0[red_idx].item() == 0.0
        # From p1's view: own hand has red
        assert t1[red_idx].item() == pytest.approx(3 / 12.0)
        assert t1[blue_idx].item() == 0.0

    def test_route_ownership_one_hot_unclaimed(self):
        """An unclaimed route encodes as [1, 0, 0] in the route block."""
        s = make_state()
        # Route block starts at offset 9 (opp_hand) + 9 (face_up) + 1 = index 19
        # Actually: hand(9) + opp_hand(1) + face_up(9) = 19, then routes at 19
        hand_block = 9
        opp_hand_block = 1
        face_up_block = 9
        route_offset = hand_block + opp_hand_block + face_up_block  # 19

        tensor = encode_state(s, perspective=0)

        link0_base = route_offset + 0 * 3
        assert tensor[link0_base].item() == 1.0      # unclaimed
        assert tensor[link0_base + 1].item() == 0.0  # not mine
        assert tensor[link0_base + 2].item() == 0.0  # not opponent's

    def test_route_ownership_one_hot_owned_by_perspective(self):
        s = make_state()
        s.route_owners[0] = 0  # perspective player owns link 0

        hand_block, opp_hand_block, face_up_block = 9, 1, 9
        route_offset = hand_block + opp_hand_block + face_up_block
        tensor = encode_state(s, perspective=0)

        link0_base = route_offset + 0 * 3
        assert tensor[link0_base].item() == 0.0      # not unclaimed
        assert tensor[link0_base + 1].item() == 1.0  # mine
        assert tensor[link0_base + 2].item() == 0.0  # not opponent's

    def test_route_ownership_one_hot_owned_by_opponent(self):
        s = make_state()
        s.route_owners[0] = 1  # opponent owns link 0

        hand_block, opp_hand_block, face_up_block = 9, 1, 9
        route_offset = hand_block + opp_hand_block + face_up_block
        tensor = encode_state(s, perspective=0)

        link0_base = route_offset + 0 * 3
        assert tensor[link0_base].item() == 0.0      # not unclaimed
        assert tensor[link0_base + 1].item() == 0.0  # not mine
        assert tensor[link0_base + 2].item() == 1.0  # opponent's

    def test_own_ticket_held_flag(self):
        """Held ticket should be encoded as (1.0, <completed>) pair."""
        s = make_state()
        ticket = CANONICAL_TICKETS[0]
        s.tickets[0] = {ticket: False}  # held but not completed

        tensor = encode_state(s, perspective=0)

        # Ticket block offset: 9+1+9+300+2+2 = 323
        ticket_offset = 9 + 1 + 9 + 300 + 2 + 2
        assert tensor[ticket_offset].item() == 1.0    # held
        assert tensor[ticket_offset + 1].item() == 0.0  # not completed

    def test_completed_ticket_flag(self):
        s = make_state()
        ticket = CANONICAL_TICKETS[0]
        s.tickets[0] = {ticket: True}  # held and completed

        tensor = encode_state(s, perspective=0)

        ticket_offset = 9 + 1 + 9 + 300 + 2 + 2
        assert tensor[ticket_offset].item() == 1.0   # held
        assert tensor[ticket_offset + 1].item() == 1.0  # completed

    def test_game_phase_one_hot_running(self):
        s = make_state()
        s.game_phase = 'running'
        tensor = encode_state(s, perspective=0)
        # Phase block is the last 3 features
        assert tensor[-3].item() == 1.0  # running
        assert tensor[-2].item() == 0.0
        assert tensor[-1].item() == 0.0

    def test_game_phase_one_hot_last_round(self):
        s = make_state()
        s.game_phase = 'last_round'
        tensor = encode_state(s, perspective=0)
        assert tensor[-3].item() == 0.0
        assert tensor[-2].item() == 1.0
        assert tensor[-1].item() == 0.0

    def test_game_phase_one_hot_finished(self):
        s = make_state()
        s.game_phase = 'finished'
        tensor = encode_state(s, perspective=0)
        assert tensor[-3].item() == 0.0
        assert tensor[-2].item() == 0.0
        assert tensor[-1].item() == 1.0


# ── Action index mapping ──────────────────────────────────────────────────────

class TestActionToIndex:
    def test_action_space_size_is_907(self):
        assert ACTION_SPACE_SIZE == 907

    def test_claim_route_index_in_range(self):
        action = Action(action_type=0, link_id=0, color='red', wilds=0)
        idx = action_to_index(action)
        assert 0 <= idx < DRAW_CARD_OFFSET

    def test_claim_route_uses_link_and_color(self):
        red_idx = COLOR_TO_IDX['red']
        blue_idx = COLOR_TO_IDX['blue']
        a_red = Action(action_type=0, link_id=5, color='red', wilds=0)
        a_blue = Action(action_type=0, link_id=5, color='blue', wilds=0)
        assert action_to_index(a_red) == CLAIM_ROUTE_OFFSET + 5 * NUM_COLORS + red_idx
        assert action_to_index(a_blue) == CLAIM_ROUTE_OFFSET + 5 * NUM_COLORS + blue_idx

    def test_different_links_get_different_indices(self):
        a0 = Action(action_type=0, link_id=0, color='red', wilds=0)
        a1 = Action(action_type=0, link_id=1, color='red', wilds=0)
        assert action_to_index(a0) != action_to_index(a1)

    def test_draw_card_index_in_correct_range(self):
        for choice in range(6):  # 0-5
            action = Action(action_type=2, card_choice=choice)
            idx = action_to_index(action)
            assert DRAW_CARD_OFFSET <= idx < DRAW_TICKETS_OFFSET

    def test_draw_card_index_uses_card_choice(self):
        for choice in range(6):
            action = Action(action_type=2, card_choice=choice)
            assert action_to_index(action) == DRAW_CARD_OFFSET + choice

    def test_draw_tickets_index_is_last(self):
        action = Action(action_type=1)
        assert action_to_index(action) == DRAW_TICKETS_OFFSET
        assert action_to_index(action) == ACTION_SPACE_SIZE - 1

    def test_all_indices_in_valid_range(self):
        actions = [
            Action(action_type=0, link_id=0, color='red', wilds=0),
            Action(action_type=0, link_id=99, color='wild', wilds=0),
            Action(action_type=2, card_choice=0),
            Action(action_type=2, card_choice=5),
            Action(action_type=1),
        ]
        for a in actions:
            idx = action_to_index(a)
            assert 0 <= idx < ACTION_SPACE_SIZE, f"Index {idx} out of range for {a}"


# ── index_to_action: inverse of action_to_index ───────────────────────────────

class TestIndexToAction:
    def test_roundtrip_claim_route(self):
        action = Action(action_type=0, link_id=7, color='blue', wilds=0)
        idx = action_to_index(action)
        recovered = index_to_action(idx, [action])
        assert recovered == action

    def test_roundtrip_draw_card(self):
        action = Action(action_type=2, card_choice=3)
        idx = action_to_index(action)
        recovered = index_to_action(idx, [action])
        assert recovered == action

    def test_roundtrip_draw_tickets(self):
        action = Action(action_type=1)
        idx = action_to_index(action)
        recovered = index_to_action(idx, [action])
        assert recovered == action

    def test_returns_none_when_no_match(self):
        action = Action(action_type=2, card_choice=0)
        idx = action_to_index(Action(action_type=2, card_choice=1))  # different action
        result = index_to_action(idx, [action])
        assert result is None

    def test_selects_correct_action_from_list(self):
        actions = [
            Action(action_type=0, link_id=0, color='red', wilds=0),
            Action(action_type=2, card_choice=3),
            Action(action_type=1),
        ]
        for a in actions:
            idx = action_to_index(a)
            assert index_to_action(idx, actions) == a


# ── Action mask ───────────────────────────────────────────────────────────────

class TestGetActionMask:
    def test_mask_shape_is_action_space_size(self):
        mask = get_action_mask([])
        assert mask.shape == (ACTION_SPACE_SIZE,)

    def test_empty_legal_actions_gives_all_false_mask(self):
        mask = get_action_mask([])
        assert not mask.any()

    def test_mask_has_exactly_n_true_values(self):
        actions = [
            Action(action_type=0, link_id=0, color='red', wilds=0),
            Action(action_type=2, card_choice=5),
            Action(action_type=1),
        ]
        mask = get_action_mask(actions)
        assert mask.sum().item() == len(actions)

    def test_mask_true_at_correct_indices(self):
        actions = [
            Action(action_type=0, link_id=3, color='blue', wilds=0),
            Action(action_type=1),
        ]
        mask = get_action_mask(actions)
        for a in actions:
            assert mask[action_to_index(a)].item()

    def test_mask_false_at_non_legal_indices(self):
        legal = [Action(action_type=1)]
        mask = get_action_mask(legal)
        tickets_idx = action_to_index(Action(action_type=1))
        other_idx = action_to_index(Action(action_type=2, card_choice=5))
        assert mask[tickets_idx].item()
        assert not mask[other_idx].item()
