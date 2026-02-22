"""Evaluate the best NEAT genome against other agents.

Usage:
    python scripts/eval_neat.py --games 50 --opponent Greedy
    python scripts/eval_neat.py --games 100 --opponent TicketFocused
    python scripts/eval_neat.py --games 50 --opponent Greedy --genome checkpoints/neat/best_genome.pkl
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from game.core import Game
from neuroevolution.manager import load_network, GAME_VERSION, MAX_MOVES_PER_GAME


def run_eval(num_games: int, opponent: str, genome_path: str = None):
    network = load_network() if genome_path is None else _load_network_from(genome_path)

    wins, total_score_diff = 0, 0
    for i in range(num_games):
        if i % 2 == 0:
            g = Game(player_types=['NEAT', opponent], version=GAME_VERSION, networks=[network, None])
            s = g.play(max_moves=MAX_MOVES_PER_GAME)
            neat_score, opp_score = s['score'][0], s['score'][1]
        else:
            g = Game(player_types=[opponent, 'NEAT'], version=GAME_VERSION, networks=[None, network])
            s = g.play(max_moves=MAX_MOVES_PER_GAME)
            neat_score, opp_score = s['score'][1], s['score'][0]

        diff = neat_score - opp_score
        total_score_diff += diff
        if diff > 0:
            wins += 1

    avg_diff = total_score_diff / num_games
    win_rate = wins / num_games
    print(f"\nNEAT vs {opponent} — {num_games} games")
    print(f"  Win rate:  {win_rate:.1%}  ({wins}/{num_games})")
    print(f"  Avg score diff: {avg_diff:+.1f}")


def _load_network_from(path: str):
    import pickle
    import neat as neat_lib
    from neuroevolution.manager import CONFIG_FILENAME
    with open(path, 'rb') as f:
        genome = pickle.load(f)
    config_path = str(Path(__file__).resolve().parent.parent / CONFIG_FILENAME)
    cnf = neat_lib.Config(neat_lib.DefaultGenome, neat_lib.DefaultReproduction,
                          neat_lib.DefaultSpeciesSet, neat_lib.DefaultStagnation,
                          config_path)
    return neat_lib.nn.FeedForwardNetwork.create(genome, cnf)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate best NEAT genome vs an agent.')
    parser.add_argument('--games', type=int, default=50, help='Number of games (default: 50)')
    parser.add_argument('--opponent', type=str, default='Greedy',
                        help='Opponent type: Greedy, TicketFocused, Random, etc. (default: Greedy)')
    parser.add_argument('--genome', type=str, default=None,
                        help='Path to genome .pkl file (default: best_genome.pkl)')
    args = parser.parse_args()
    run_eval(args.games, args.opponent, args.genome)
