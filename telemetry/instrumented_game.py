"""Instrumented game loop that captures telemetry without modifying core.py.

Re-implements Game.play() externally while recording pre/post state each turn.
"""
import uuid
from typing import List, Dict

from game.core import Game
from game.enums import GameState, ActionDecision
from telemetry.collector import TelemetryCollector


def run_instrumented_game(
    player_types: List[str],
    version: str,
    collector: TelemetryCollector,
    max_moves: int = 0,
    networks=None,
) -> Dict:
    """Run a single game with full telemetry capture.

    Mirrors the logic of Game.play() but captures state before/after each turn
    to record decisions and outcomes.

    Returns:
        Game stats dictionary (same format as Game.play()).
    """
    game = Game(player_types, version, networks=networks)
    game_id = str(uuid.uuid4())
    collector.start_game(game_id, version, player_types)

    # === Initialization (mirrors core.py lines 97-104) ===
    game.ticket_deck.set_ticket_pile_num_adapter()
    game.game_state = GameState.RUNNING

    for player in game.players:
        player.draw_initial_train_cards(game.config.NUM_TRAIN_CARDS_DEALT_INIT)
        player.draw_tickets(
            num_tickets=game.config.NUM_TICKETS_DEALT_INIT,
            min_keep=game.config.MIN_TICKETS_KEPT_INIT,
        )

    turn_number = 0

    # === Main loop (mirrors core.py lines 106-136) ===
    while game.game_state != GameState.FINISHED:
        if max_moves and sum(game.stats['total_moves']) >= max_moves:
            break

        current_player = game.players[game.current_player_id]

        # Capture pre-turn state
        pre_state = _capture_player_state(current_player, game)

        game.stats['total_moves'][game.current_player_id] += 1
        move_completed = current_player.play_turn()

        # Capture post-turn state
        post_state = _capture_player_state(current_player, game)
        action = _infer_action(pre_state, post_state)
        detail = _compute_decision_detail(pre_state, post_state)

        collector.record_turn(
            game_id=game_id,
            turn_number=turn_number,
            player_id=current_player.player_id,
            player_type=player_types[current_player.player_id],
            action=action,
            state_snapshot=pre_state,
            decision_detail=detail,
            move_valid=move_completed,
        )

        if move_completed:
            game.stats['completed_moves'][current_player.player_id] += 1
        else:
            game.stats['invalid_moves'][current_player.player_id] += 1

        # State transitions (mirrors core.py logic)
        if game.game_state == GameState.RUNNING:
            if game.last_round_condition(current_player):
                game.last_player = current_player
                game.game_state = GameState.LAST_ROUND
            game.move_to_next_player()
        elif game.game_state == GameState.LAST_ROUND:
            if current_player is game.last_player:
                game.game_state = GameState.FINISHED
            else:
                game.move_to_next_player()

        turn_number += 1

    # === End-game scoring (mirrors core.py lines 138-156) ===
    game.score_player_tickets()
    game.score_longest_path()
    game.determine_winner()

    game.stats['completed_tickets'] = [sum(p.tickets.values()) for p in game.players]
    game.stats['total_tickets'] = [len(p.tickets) for p in game.players]
    game.stats['trains_remaining'] = [p.trains_remaining for p in game.players]
    game.stats['longest_path_owner'] = [p.longest_path for p in game.players]
    game.stats['score'] = [p.score for p in game.players]
    game.stats['claimed_routes'] = [game.board.count_claimed_routes(p.color) for p in game.players]

    collector.record_game_result(game_id, game.stats)

    return game.stats


def _capture_player_state(player, game) -> dict:
    """Snapshot relevant player + game state for telemetry."""
    return {
        'hand': dict(player.hand),
        'score': player.score,
        'trains_remaining': player.trains_remaining,
        'tickets_held': len(player.tickets),
        'tickets_completed': sum(player.tickets.values()),
    }


def _infer_action(pre: dict, post: dict) -> int:
    """Infer what action was taken by comparing pre/post state."""
    trains_delta = pre['trains_remaining'] - post['trains_remaining']
    tickets_delta = post['tickets_held'] - pre['tickets_held']
    hand_delta = sum(post['hand'].values()) - sum(pre['hand'].values())

    if trains_delta > 0:
        return ActionDecision.CLAIM_ROUTE.value
    elif tickets_delta > 0:
        return ActionDecision.DRAW_TICKETS.value
    elif hand_delta > 0:
        return ActionDecision.DRAW_CARDS.value
    else:
        return ActionDecision.SKIP.value


def _compute_decision_detail(pre: dict, post: dict) -> dict:
    """Compute what specifically changed between pre and post state."""
    hand_before = pre['hand']
    hand_after = post['hand']
    cards_gained = {}
    cards_spent = {}

    all_colors = set(list(hand_before.keys()) + list(hand_after.keys()))
    for color in all_colors:
        delta = hand_after.get(color, 0) - hand_before.get(color, 0)
        if delta > 0:
            cards_gained[color] = delta
        elif delta < 0:
            cards_spent[color] = abs(delta)

    return {
        'cards_gained': cards_gained,
        'cards_spent': cards_spent,
        'trains_used': pre['trains_remaining'] - post['trains_remaining'],
        'score_gained': post['score'] - pre['score'],
        'tickets_gained': post['tickets_held'] - pre['tickets_held'],
    }
