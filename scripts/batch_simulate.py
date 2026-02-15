"""Batch simulation runner for comparing agent strategies.

Usage:
    python scripts/batch_simulate.py --games 1000 --players Greedy TicketFocused CardHoarder Blocker --version USA
    python scripts/batch_simulate.py --games 500 --players Random Random Greedy TicketFocused --parallel 4
    python scripts/batch_simulate.py --games 100 --players Greedy Random --telemetry telemetry.db
"""
import argparse
import json
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path when running as script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict


def run_single_game(args_tuple) -> Dict:
    """Run one game. Top-level function for ProcessPoolExecutor pickling."""
    player_types, version, max_moves, game_index = args_tuple
    from game.core import Game
    game = Game(player_types, version)
    stats = game.play(max_moves=max_moves)
    return {
        'game_index': game_index,
        'player_types': player_types,
        'stats': stats,
    }


def run_batch(
    num_games: int,
    player_types: List[str],
    version: str = 'USA',
    max_moves: int = 1000,
    parallel: int = 1,
    telemetry_db: str = None,
    quiet: bool = False,
) -> Dict:
    """Run N games and aggregate results.

    Args:
        num_games: Number of games to simulate.
        player_types: List of player type strings (2-5 players).
        version: Game version (USA, Europe).
        max_moves: Max moves per game (safety valve).
        parallel: Number of parallel workers (1 = sequential).
        telemetry_db: Optional path to SQLite telemetry database.
        quiet: Suppress progress output.

    Returns:
        Aggregated statistics dictionary.
    """
    results = []
    start_time = time.time()

    if parallel > 1 and telemetry_db is None:
        args_list = [(player_types, version, max_moves, i) for i in range(num_games)]
        with ProcessPoolExecutor(max_workers=parallel) as executor:
            futures = [executor.submit(run_single_game, args) for args in args_list]
            for future in as_completed(futures):
                results.append(future.result())
                if not quiet and len(results) % 100 == 0:
                    print(f"Completed {len(results)}/{num_games} games...")
    else:
        collector = None
        if telemetry_db:
            from telemetry.collector import TelemetryCollector
            collector = TelemetryCollector(telemetry_db)

        from game.core import Game

        for i in range(num_games):
            if collector:
                from telemetry.instrumented_game import run_instrumented_game
                stats = run_instrumented_game(player_types, version, collector, max_moves)
            else:
                game = Game(player_types, version)
                stats = game.play(max_moves=max_moves)

            results.append({'game_index': i, 'player_types': player_types, 'stats': stats})

            if not quiet and (i + 1) % 100 == 0:
                print(f"Completed {i + 1}/{num_games} games...")
                if collector:
                    collector.flush()

        if collector:
            collector.close()

    elapsed = time.time() - start_time

    summary = aggregate_results(results, player_types)
    summary['elapsed_seconds'] = round(elapsed, 2)
    summary['games_per_second'] = round(num_games / elapsed, 1) if elapsed > 0 else 0

    return summary


def aggregate_results(results: List[Dict], player_types: List[str]) -> Dict:
    """Compute summary statistics from game results."""
    num_players = len(player_types)
    num_games = len(results)

    wins = [0] * num_players
    total_scores = [0] * num_players
    total_completed_tickets = [0] * num_players
    total_tickets = [0] * num_players
    total_claimed_routes = [0] * num_players
    score_lists = [[] for _ in range(num_players)]

    for result in results:
        stats = result['stats']
        scores = stats['score']
        winner_id = scores.index(max(scores))
        wins[winner_id] += 1

        for pid in range(num_players):
            total_scores[pid] += scores[pid]
            score_lists[pid].append(scores[pid])
            total_completed_tickets[pid] += stats['completed_tickets'][pid]
            total_tickets[pid] += stats['total_tickets'][pid]
            total_claimed_routes[pid] += stats['claimed_routes'][pid]

    summary = {
        'num_games': num_games,
        'player_types': player_types,
        'per_player': [],
    }

    for pid in range(num_players):
        avg_score = total_scores[pid] / num_games
        scores = score_lists[pid]
        std_dev = (sum((s - avg_score) ** 2 for s in scores) / num_games) ** 0.5

        ticket_completion_rate = (
            total_completed_tickets[pid] / total_tickets[pid]
            if total_tickets[pid] > 0 else 0.0
        )

        summary['per_player'].append({
            'player_id': pid,
            'player_type': player_types[pid],
            'win_rate': round(wins[pid] / num_games, 4),
            'wins': wins[pid],
            'avg_score': round(avg_score, 1),
            'score_std_dev': round(std_dev, 1),
            'min_score': min(scores),
            'max_score': max(scores),
            'avg_completed_tickets': round(total_completed_tickets[pid] / num_games, 2),
            'avg_total_tickets': round(total_tickets[pid] / num_games, 2),
            'ticket_completion_rate': round(ticket_completion_rate, 4),
            'avg_claimed_routes': round(total_claimed_routes[pid] / num_games, 2),
        })

    return summary


def print_summary(summary: Dict):
    """Pretty-print batch simulation results."""
    print(f"\n{'=' * 70}")
    print(f"BATCH SIMULATION RESULTS ({summary['num_games']} games)")
    print(f"{'=' * 70}")
    print(f"Time: {summary['elapsed_seconds']}s ({summary['games_per_second']} games/sec)")
    print(f"Players: {summary['player_types']}")
    print()

    for stats in summary['per_player']:
        print(f"--- {stats['player_type']} (Player {stats['player_id']}) ---")
        print(f"  Win rate:              {stats['win_rate']:.1%}")
        print(f"  Avg score:             {stats['avg_score']:.1f} (std: {stats['score_std_dev']:.1f})")
        print(f"  Score range:           [{stats['min_score']}, {stats['max_score']}]")
        print(f"  Ticket completion:     {stats['ticket_completion_rate']:.1%}")
        print(f"  Avg tickets held:      {stats['avg_total_tickets']:.1f}")
        print(f"  Avg tickets completed: {stats['avg_completed_tickets']:.1f}")
        print(f"  Avg routes claimed:    {stats['avg_claimed_routes']:.1f}")
        print()


def main():
    parser = argparse.ArgumentParser(description='Batch simulate Ticket to Ride games')
    parser.add_argument('--games', type=int, default=100, help='Number of games to simulate')
    parser.add_argument('--players', nargs='+',
                        default=['Random', 'Random', 'Greedy', 'TicketFocused'],
                        help='Player types (2-5). Options: Random, Greedy, TicketFocused, CardHoarder, Blocker')
    parser.add_argument('--version', default='USA', help='Game version (USA, Europe)')
    parser.add_argument('--max-moves', type=int, default=1000, help='Max moves per game')
    parser.add_argument('--parallel', type=int, default=1, help='Number of parallel workers')
    parser.add_argument('--telemetry', type=str, default=None, help='SQLite telemetry DB path')
    parser.add_argument('--output', type=str, default=None, help='JSON output file path')
    parser.add_argument('--quiet', action='store_true', help='Suppress progress output')

    args = parser.parse_args()

    if not 2 <= len(args.players) <= 5:
        print("Error: Need 2-5 players", file=sys.stderr)
        sys.exit(1)

    summary = run_batch(
        num_games=args.games,
        player_types=args.players,
        version=args.version,
        max_moves=args.max_moves,
        parallel=args.parallel,
        telemetry_db=args.telemetry,
        quiet=args.quiet,
    )

    print_summary(summary)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()
