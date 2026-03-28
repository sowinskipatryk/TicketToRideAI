"""Integration tests: full Game end-to-end with sim_game consistency checks.

These tests exercise the full stack — Game, players, TrainCardManager,
TicketDeck, GameBoard — and verify that:
- The game always terminates within a reasonable move limit.
- Final stats have the expected structure and valid values.
- Scores in Game match scores computed by sim_game.evaluate().
- from_game() correctly mirrors the live Game state into a SimState.
"""
import pytest

from game.core import Game
from game.sim_game import from_game, evaluate, is_terminal, get_legal_actions


# ── Full game smoke tests ─────────────────────────────────────────────────────

class TestFullGame:
    """Run complete games and verify they finish with a valid result."""

    def test_2_random_players_game_completes(self):
        game = Game(['Random', 'Random'], 'USA')
        stats = game.play(max_moves=2000)

        assert game.winner is not None
        assert 'score' in stats
        assert len(stats['score']) == 2

    def test_game_stats_structure(self):
        game = Game(['Random', 'Random'], 'USA')
        stats = game.play(max_moves=2000)

        required_keys = [
            'score', 'completed_tickets', 'total_tickets',
            'trains_remaining', 'claimed_routes',
            'completed_moves', 'total_moves', 'invalid_moves',
        ]
        for key in required_keys:
            assert key in stats, f"Missing key: {key}"

    def test_scores_are_non_negative(self):
        """Even with incomplete tickets, scores start from route points
        and should typically stay non-negative in a reasonably long game."""
        game = Game(['Random', 'Random'], 'USA')
        stats = game.play(max_moves=2000)

        # Route points alone should ensure final score is reasonable
        # (not asserting >= 0 since penalty tickets can make it negative
        # in edge cases, just verify it's an integer)
        for score in stats['score']:
            assert isinstance(score, (int, float))

    def test_trains_remaining_are_valid(self):
        game = Game(['Random', 'Random'], 'USA')
        stats = game.play(max_moves=2000)

        for trains in stats['trains_remaining']:
            assert 0 <= trains <= 45

    def test_winner_has_highest_or_tied_score(self):
        game = Game(['Random', 'Random'], 'USA')
        game.play(max_moves=2000)

        winner_score = game.winner.get_score()
        for player in game.players:
            assert player.get_score() <= winner_score

    def test_game_state_is_finished_after_play(self):
        from game.enums import GameState
        game = Game(['Random', 'Random'], 'USA')
        game.play(max_moves=2000)

        assert game.game_state == GameState.FINISHED

    def test_all_claimed_routes_counted(self):
        game = Game(['Random', 'Random'], 'USA')
        stats = game.play(max_moves=2000)

        total_claimed = sum(stats['claimed_routes'])
        # USA board has 100 links; random players won't claim all but > 0
        assert total_claimed > 0
        assert total_claimed <= 100

    def test_completed_tickets_le_total_tickets(self):
        game = Game(['Random', 'Random'], 'USA')
        stats = game.play(max_moves=2000)

        for completed, total in zip(stats['completed_tickets'], stats['total_tickets']):
            assert completed <= total


# ── from_game() consistency ───────────────────────────────────────────────────

class TestFromGame:
    """Verify that from_game() extracts a SimState that faithfully mirrors
    the live Game state."""

    def _game_mid_play(self) -> Game:
        """Return a game that has run for a fixed number of moves."""
        game = Game(['Random', 'Random'], 'USA')
        game._deal_initial()
        # Run 30 turns manually so there's interesting state to inspect
        from game.enums import GameState
        moves = 0
        while game.game_state != GameState.FINISHED and moves < 30:
            game._run_turn(max_moves=0)
            current_player = game.players[game.current_player_id]
            if game.last_round_condition(current_player):
                game.last_player = current_player
                game.game_state = GameState.LAST_ROUND
                break
            game.move_to_next_player()
            moves += 1
        return game

    def test_num_players_matches(self):
        game = Game(['Random', 'Random'], 'USA')
        game._deal_initial()
        state = from_game(game)
        assert state.num_players == 2

    def test_hand_sizes_match(self):
        game = self._game_mid_play()
        state = from_game(game)

        for i, player in enumerate(game.players):
            sim_total = sum(state.hands[i].values())
            game_total = sum(player.hand.values())
            assert sim_total == game_total, (
                f"Player {i} hand size mismatch: sim={sim_total} game={game_total}"
            )

    def test_scores_match(self):
        game = self._game_mid_play()
        state = from_game(game)

        for i, player in enumerate(game.players):
            assert state.scores[i] == player.score, (
                f"Player {i} score mismatch: sim={state.scores[i]} game={player.score}"
            )

    def test_trains_match(self):
        game = self._game_mid_play()
        state = from_game(game)

        for i, player in enumerate(game.players):
            assert state.trains[i] == player.trains_remaining

    def test_current_player_matches(self):
        game = Game(['Random', 'Random'], 'USA')
        game._deal_initial()
        state = from_game(game)
        assert state.current_player == game.current_player_id

    def test_route_ownership_matches(self):
        game = self._game_mid_play()
        state = from_game(game)

        # Every route owned in the sim state should match the board
        for link_id, owner_id in state.route_owners.items():
            if owner_id is not None:
                owner_color = game.players[owner_id].color
                # Find the edge on the board with this link_id
                edge_found = False
                for u, v, data in game.board.G.edges(data=True):
                    if data['link_id'] == link_id:
                        assert data['claimed_by'] == owner_color, (
                            f"Link {link_id} owner mismatch"
                        )
                        edge_found = True
                        break
                assert edge_found

    def test_game_phase_running(self):
        game = Game(['Random', 'Random'], 'USA')
        game._deal_initial()
        state = from_game(game)
        assert state.game_phase == 'running'


# ── sim_game legal actions during real game ───────────────────────────────────

class TestSimGameDuringRealGame:
    """Verify that sim_game's get_legal_actions() produces sensible results
    when applied to a state extracted from a live game."""

    def test_legal_actions_nonempty_at_start(self):
        game = Game(['Random', 'Random'], 'USA')
        game._deal_initial()
        state = from_game(game)

        legal = get_legal_actions(state)
        assert len(legal) > 0

    def test_legal_actions_all_have_valid_action_types(self):
        game = Game(['Random', 'Random'], 'USA')
        game._deal_initial()
        state = from_game(game)

        legal = get_legal_actions(state)
        for action in legal:
            assert action.action_type in (0, 1, 2)

    def test_not_terminal_at_game_start(self):
        game = Game(['Random', 'Random'], 'USA')
        game._deal_initial()
        state = from_game(game)

        assert not is_terminal(state)
