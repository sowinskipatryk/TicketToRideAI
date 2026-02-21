"""NEAT network training manager."""
import copy
import neat
import os
import pickle
import time

# Try to load config, fall back to defaults
try:
    from game.config_loader import load_config
    _config = load_config()
    _neat_config = _config.get('neat', {})
    CONFIG_FILENAME = _neat_config.get('config_filename', 'neuroevolution/neat_config.txt')
    GENOME_FILENAME = _neat_config.get('genome_filename', 'best_genome.pkl')
    PLAYERS_NUM = _neat_config.get('players_num', 2)
    GAME_VERSION = _neat_config.get('game_version', 'USA')
    NUM_GENERATIONS = _neat_config.get('num_generations', 20)
    MAX_MOVES_PER_GAME = _neat_config.get('max_moves_per_game', 1000)
except (ImportError, Exception):
    # Default values if config loading fails
    CONFIG_FILENAME = "neat_config.txt"
    GENOME_FILENAME = 'best_genome.pkl'
    PLAYERS_NUM = 2
    GAME_VERSION = 'USA'
    NUM_GENERATIONS = 20
    MAX_MOVES_PER_GAME = 1000


class TrainingReporter(neat.reporting.BaseReporter):
    """Compact one-line-per-generation reporter.

    Shows running best instead of average fitness — average is always 0.0
    in 2-player zero-sum pairing (one genome's gain = opponent's loss).

    Also runs 2 calibration games per generation against GreedyRouteAgent
    to track absolute improvement independent of opponent strength.
    """

    def __init__(self, num_generations: int):
        self.num_generations = num_generations
        self.generation = 0
        self._gen_start = None
        self._best_ever = float('-inf')
        self.best_genome = None             # copy of best genome seen, frozen at time of best fitness
        self.calibration_scores: list = []  # vs Greedy per generation

    def start_generation(self, generation):
        self.generation = generation
        self._gen_start = time.time()

    def post_evaluate(self, config, population, species_set, best_genome):
        elapsed = time.time() - self._gen_start
        if best_genome.fitness > self._best_ever:
            self._best_ever = best_genome.fitness
            self.best_genome = copy.deepcopy(best_genome)
        num_species = len(species_set.species)
        best_nodes = len(best_genome.nodes)
        calibration = _calibrate(best_genome, config)
        self.calibration_scores.append(calibration)
        print(
            f"Gen {self.generation + 1:2d}/{self.num_generations} | "
            f"Best: {best_genome.fitness:+7.1f} | "
            f"All-time: {self._best_ever:+7.1f} | "
            f"vs Greedy: {calibration:+6.1f} | "
            f"Species: {num_species:2d} | "
            f"Nodes: {best_nodes:3d} | "
            f"Time: {elapsed:.1f}s"
        )

    def info(self, msg):
        pass  # suppress NEAT's internal info messages


def _calibrate(genome, config) -> float:
    """Measure absolute strength: play best genome vs GreedyRouteAgent (2 games, swapped sides).

    Self-play fitness deflates as all genomes improve together. This gives a
    stable external reference — if vs-Greedy score rises over generations,
    the population is genuinely getting better.
    """
    from game.core import Game
    network = neat.nn.FeedForwardNetwork.create(genome, config)

    total = 0
    # Game 1: NEAT goes first
    g1 = Game(player_types=['NEAT', 'Greedy'], version=GAME_VERSION,
              networks=[network, None])
    s1 = g1.play(max_moves=MAX_MOVES_PER_GAME)
    total += s1['score'][0] - s1['score'][1]

    # Game 2: Greedy goes first
    g2 = Game(player_types=['Greedy', 'NEAT'], version=GAME_VERSION,
              networks=[None, network])
    s2 = g2.play(max_moves=MAX_MOVES_PER_GAME)
    total += s2['score'][1] - s2['score'][0]

    return total / 2



def load_network():
    if os.path.exists(GENOME_FILENAME):
        with open(GENOME_FILENAME, "rb") as f:
            genome = pickle.load(f)

        config_path = os.path.join(os.path.dirname(__file__), CONFIG_FILENAME)
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_path)
        return neat.nn.FeedForwardNetwork.create(genome, config)
    else:
        raise FileNotFoundError(
            'The genome file was not found! Train a network first or choose a different player type.'
        )


def eval_genomes(genomes, config):
    from game.core import Game

    for i in range(0, len(genomes) - (len(genomes) % PLAYERS_NUM), PLAYERS_NUM):
        genome_pairs = [genomes[i + j] for j in range(PLAYERS_NUM)]
        networks = [neat.nn.FeedForwardNetwork.create(genome, config)
                    for _, genome in genome_pairs]

        # Accumulate score differences over two games with swapped starting positions.
        # This averages out first-player advantage and random card luck.
        score_diffs = [0] * PLAYERS_NUM
        for rotation in range(PLAYERS_NUM):
            rotated_networks = networks[rotation:] + networks[:rotation]
            game = Game(player_types=['NEAT'] * PLAYERS_NUM, version=GAME_VERSION,
                        networks=rotated_networks)
            stats = game.play(max_moves=MAX_MOVES_PER_GAME)
            rotated_scores = stats['score']
            # Map rotated positions back to original genome indices
            for pos in range(PLAYERS_NUM):
                original_j = (pos - rotation) % PLAYERS_NUM
                opp_score = max(rotated_scores[k] for k in range(PLAYERS_NUM) if k != pos)
                score_diffs[original_j] += rotated_scores[pos] - opp_score

        for j in range(PLAYERS_NUM):
            _, genome = genome_pairs[j]
            genome.fitness = score_diffs[j] / PLAYERS_NUM


def save_genome(genome):
    with open(GENOME_FILENAME, 'wb') as file:
        pickle.dump(genome, file)


def _save_plot(stats, reporter: TrainingReporter, filename='neat_training.png'):
    try:
        import matplotlib.pyplot as plt

        best_per_gen = [g.fitness for g in stats.most_fit_genomes]
        running_best = []
        current_best = float('-inf')
        for f in best_per_gen:
            current_best = max(current_best, f)
            running_best.append(current_best)
        generations = range(1, len(best_per_gen) + 1)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Top panel: self-play fitness
        ax1.plot(generations, best_per_gen, 'b-o', label='Best this gen', markersize=4, alpha=0.6)
        ax1.plot(generations, running_best, 'g-', label='All-time best', linewidth=2)
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.set_ylabel('Self-play fitness')
        ax1.set_title('NEAT Training Progress')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Bottom panel: calibration vs GreedyRouteAgent
        if reporter.calibration_scores:
            ax2.plot(generations, reporter.calibration_scores, 'r-o',
                     label='vs Greedy (absolute)', markersize=4)
            ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax2.set_ylabel('Score diff vs Greedy')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        ax2.set_xlabel('Generation')
        fig.tight_layout()
        fig.savefig(filename, dpi=100)
        plt.close(fig)
        print(f"Plot saved: {filename}")
    except ImportError:
        pass


def run_neat(resume_checkpoint: str = None):
    config_path = os.path.join(os.path.dirname(__file__), CONFIG_FILENAME)
    cnf = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                      neat.DefaultSpeciesSet, neat.DefaultStagnation,
                      config_path)

    if resume_checkpoint:
        print(f"Resuming from checkpoint: {resume_checkpoint}\n")
        population = neat.Checkpointer.restore_checkpoint(resume_checkpoint)
    else:
        population = neat.Population(cnf)

    print(f"NEAT Training | pop={cnf.pop_size} | generations={NUM_GENERATIONS} | "
          f"players={PLAYERS_NUM} | version={GAME_VERSION}\n")

    reporter = TrainingReporter(NUM_GENERATIONS)
    stats = neat.StatisticsReporter()
    population.add_reporter(reporter)
    population.add_reporter(stats)
    population.add_reporter(neat.Checkpointer(generation_interval=1))

    population.run(eval_genomes, NUM_GENERATIONS)
    best = reporter.best_genome
    save_genome(best)

    print(f"\nTraining complete | Best fitness: {reporter._best_ever:+.1f} | Saved: {GENOME_FILENAME}")
    _save_plot(stats, reporter)
