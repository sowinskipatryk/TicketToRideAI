"""NEAT network training manager."""
import copy
import gzip
import json
import neat
import os
import pickle
import random
import time

CHECKPOINT_DIR = 'checkpoints/neat'


class _Checkpointer(neat.Checkpointer):
    """Checkpointer that saves to CHECKPOINT_DIR with 1-indexed generation numbers."""

    def save_checkpoint(self, config, population, species_set, generation):
        filename = f'{self.filename_prefix}{generation + 1}'
        print(f"Saving checkpoint to {filename}")
        with gzip.open(filename, 'w', compresslevel=5) as f:
            data = (generation, config, population, species_set, random.getstate())
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

# Try to load config, fall back to defaults
try:
    from game.config_loader import load_config
    _config = load_config()
    _neat_config = _config.get('neat', {})
    CONFIG_FILENAME = _neat_config.get('config_filename', 'neuroevolution/neat_config.txt')
    GENOME_FILENAME = _neat_config.get('genome_filename', os.path.join(CHECKPOINT_DIR, 'best_genome.pkl'))
    PLAYERS_NUM = _neat_config.get('players_num', 2)
    GAME_VERSION = _neat_config.get('game_version', 'USA')
    NUM_GENERATIONS = _neat_config.get('num_generations', 20)
    MAX_MOVES_PER_GAME = _neat_config.get('max_moves_per_game', 1000)
except (ImportError, Exception):
    # Default values if config loading fails
    CONFIG_FILENAME = "neat_config.txt"
    GENOME_FILENAME = os.path.join(CHECKPOINT_DIR, 'best_genome.pkl')
    PLAYERS_NUM = 2
    GAME_VERSION = 'USA'
    NUM_GENERATIONS = 100
    MAX_MOVES_PER_GAME = 1000


class TrainingReporter(neat.reporting.BaseReporter):
    """Compact one-line-per-generation reporter.

    Fitness = 0.5 * score_diff_vs_NEAT + 0.5 * score_diff_vs_TicketFocused,
    so the population average is negative (TicketFocused beats most genomes).
    Best fitness per generation is shown, not average.

    Also calibrates top-3 genomes vs TicketFocused (10 games) each generation
    to track absolute improvement and select the best genome to save.
    """

    def __init__(self, num_generations: int, log_path: str, start_generation: int = 0):
        self.num_generations = num_generations
        self.log_path = log_path
        self.generation = 0
        self._gen_start = None
        self._best_calibration = float('-inf')  # guards best_genome: immune to early opponent inflation
        self.best_genome = None  # genome with highest vs-Greedy calibration score seen this session
        self.training_log: list = []
        if start_generation > 0 and os.path.exists(log_path):
            with open(log_path) as f:
                full_log = json.load(f)
            # Keep only entries up to and including the checkpoint generation,
            # discarding any entries from generations that ran after the checkpoint.
            self.training_log = [e for e in full_log if e['generation'] <= start_generation]
            if self.training_log:
                self._best_calibration = max(e['best_calibration'] for e in self.training_log)

    def start_generation(self, generation):
        self.generation = generation
        self._gen_start = time.time()

    def post_evaluate(self, config, population, species_set, best_genome):
        elapsed = time.time() - self._gen_start
        num_species = len(species_set.species)
        best_nodes = len(best_genome.nodes)

        # Calibrate top 3 by self-play fitness; pick the best vs TicketFocused among them.
        # This guards against the case where the self-play winner got lucky against a weak opponent.
        top3 = sorted(population.values(), key=lambda g: g.fitness, reverse=True)[:3]
        best_cal_score = float('-inf')
        best_cal_genome = None
        for g in top3:
            cal = _calibrate(g, config)
            if cal > best_cal_score:
                best_cal_score = cal
                best_cal_genome = g

        if best_cal_score > self._best_calibration:
            self._best_calibration = best_cal_score
            self.best_genome = copy.deepcopy(best_cal_genome)
            save_genome(self.best_genome)
        self.training_log.append({
            'generation': self.generation + 1,
            'best_fitness': best_genome.fitness,
            'best_calibration': round(self._best_calibration, 2),
            'vs_ticketFocused': round(best_cal_score, 2),
        })
        with open(self.log_path, 'w') as f:
            json.dump(self.training_log, f, indent=2)
        print(
            f"Gen {self.generation + 1:2d}/{self.num_generations} | "
            f"Best: {best_genome.fitness:+7.1f} | "
            f"vs TicketFocused: {best_cal_score:+6.1f} | "
            f"Best vs TicketFocused: {self._best_calibration:+6.1f} | "
            f"Species: {num_species:2d} | "
            f"Nodes: {best_nodes:3d} | "
            f"Time: {elapsed:.1f}s"
        )

    def info(self, msg):
        pass  # suppress NEAT's internal info messages


def _calibrate(genome, config, num_games: int = 20) -> float:
    """Measure absolute strength vs TicketFocused (num_games total, alternating sides).

    Self-play fitness deflates as all genomes improve together. This gives a
    stable external reference — if vs-TicketFocused score rises over generations,
    the population is genuinely getting better.

    num_games=10 (5 per side) reduces variance enough to see a real trend.
    """
    from game.core import Game
    network = neat.nn.FeedForwardNetwork.create(genome, config)

    total = 0
    for i in range(num_games):
        if i % 2 == 0:
            g = Game(player_types=['NEAT', 'TicketFocused'], version=GAME_VERSION,
                     networks=[network, None])
            s = g.play(max_moves=MAX_MOVES_PER_GAME)
            total += s['score'][0] - s['score'][1]
        else:
            g = Game(player_types=['TicketFocused', 'NEAT'], version=GAME_VERSION,
                     networks=[None, network])
            s = g.play(max_moves=MAX_MOVES_PER_GAME)
            total += s['score'][1] - s['score'][0]

    return total / num_games



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

    for _, genome in genomes:
        genome.fitness = 0.0  # default for any unpaired genome (when pop_size is odd)

    # Phase 1: NEAT-vs-NEAT pairing (two rotations to average out first-player advantage).
    neat_scores = {}
    for i in range(0, len(genomes) - (len(genomes) % PLAYERS_NUM), PLAYERS_NUM):
        genome_pairs = [genomes[i + j] for j in range(PLAYERS_NUM)]
        networks = [neat.nn.FeedForwardNetwork.create(genome, config)
                    for _, genome in genome_pairs]
        score_diffs = [0] * PLAYERS_NUM
        for rotation in range(PLAYERS_NUM):
            rotated_networks = networks[rotation:] + networks[:rotation]
            game = Game(player_types=['NEAT'] * PLAYERS_NUM, version=GAME_VERSION,
                        networks=rotated_networks)
            stats = game.play(max_moves=MAX_MOVES_PER_GAME)
            rotated_scores = stats['score']
            for pos in range(PLAYERS_NUM):
                original_j = (pos - rotation) % PLAYERS_NUM
                opp_score = max(rotated_scores[k] for k in range(PLAYERS_NUM) if k != pos)
                score_diffs[original_j] += rotated_scores[pos] - opp_score
        for j in range(PLAYERS_NUM):
            gid, _ = genome_pairs[j]
            neat_scores[gid] = score_diffs[j] / PLAYERS_NUM

    # Phase 2: Every genome plays 1 game vs TicketFocused (alternating sides across population).
    tf_scores = {}
    for idx, (gid, genome) in enumerate(genomes):
        network = neat.nn.FeedForwardNetwork.create(genome, config)
        if idx % 2 == 0:
            g = Game(player_types=['NEAT', 'TicketFocused'], version=GAME_VERSION,
                     networks=[network, None])
            s = g.play(max_moves=MAX_MOVES_PER_GAME)
            tf_scores[gid] = s['score'][0] - s['score'][1]
        else:
            g = Game(player_types=['TicketFocused', 'NEAT'], version=GAME_VERSION,
                     networks=[None, network])
            s = g.play(max_moves=MAX_MOVES_PER_GAME)
            tf_scores[gid] = s['score'][1] - s['score'][0]

    # Combine: 50% NEAT-vs-NEAT + 50% vs TicketFocused.
    for gid, genome in genomes:
        genome.fitness = 0.5 * neat_scores.get(gid, 0.0) + 0.5 * tf_scores.get(gid, 0.0)


def save_genome(genome):
    os.makedirs(os.path.dirname(GENOME_FILENAME), exist_ok=True)
    with open(GENOME_FILENAME, 'wb') as file:
        pickle.dump(genome, file)


def _save_plot(training_log: list, filename: str):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        if len(training_log) < 2:
            return

        generations = [e['generation'] for e in training_log]
        best_per_gen = [e['best_fitness'] for e in training_log]
        vs_greedy = [e['vs_ticketFocused'] for e in training_log]
        best_calibration = [e['best_calibration'] for e in training_log]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        ax1.plot(generations, best_per_gen, 'b-o', label='Best this gen', markersize=4, alpha=0.6)
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.set_ylabel('Self-play fitness')
        ax1.set_title('NEAT Training Progress')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(generations, vs_greedy, 'r-o', label='vs TicketFocused', markersize=4, alpha=0.6)
        ax2.plot(generations, best_calibration, 'g-', label='Best vs TicketFocused ever', linewidth=2)
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_ylabel('Score diff vs TicketFocused')
        ax2.set_xlabel('Generation')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

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
        if not os.path.exists(resume_checkpoint):
            candidate = os.path.join(CHECKPOINT_DIR, resume_checkpoint)
            if os.path.exists(candidate):
                resume_checkpoint = candidate
        print(f"Resuming from checkpoint: {resume_checkpoint}\n")
        population = neat.Checkpointer.restore_checkpoint(resume_checkpoint)
        # Skip the already-completed checkpoint generation so NEAT starts fresh from the next one.
        population.generation += 1
        start_generation = population.generation  # 1-indexed: e.g. 90 for neat-checkpoint-90
        generations_to_run = max(1, NUM_GENERATIONS - population.generation)
    else:
        start_generation = 0  # fresh start: don't load any existing log
        population = neat.Population(cnf)
        generations_to_run = NUM_GENERATIONS

    print(f"NEAT Training | population={cnf.pop_size} | generations={NUM_GENERATIONS} | "
          f"players={PLAYERS_NUM} | version={GAME_VERSION}\n")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_prefix = os.path.join(CHECKPOINT_DIR, 'neat-checkpoint-')
    log_path = os.path.join(CHECKPOINT_DIR, 'training_log.json')
    plot_path = os.path.join(CHECKPOINT_DIR, 'training_progress.png')

    reporter = TrainingReporter(NUM_GENERATIONS, log_path, start_generation)
    checkpointer = _Checkpointer(generation_interval=5, time_interval_seconds=None, filename_prefix=checkpoint_prefix)
    # Align last_generation_checkpoint so checkpoints land on exact multiples of the interval
    # (display gen 5, 10, 15, ..., 100) regardless of resume point.
    # Internal gen = display gen - 1. For the first desired display checkpoint after start_generation:
    #   next_display = (start_generation // interval + 1) * interval
    #   last = (next_display - 1) - interval
    interval = checkpointer.generation_interval
    if start_generation > 0:
        next_display = (start_generation // interval + 1) * interval
        checkpointer.last_generation_checkpoint = (next_display - 1) - interval
    else:
        checkpointer.last_generation_checkpoint = -1
    population.add_reporter(reporter)
    population.add_reporter(checkpointer)

    population.run(eval_genomes, generations_to_run)

    print(f"\nTraining complete | Best vs TicketFocused: {reporter._best_calibration:+.1f} | Saved: {GENOME_FILENAME}")
    _save_plot(reporter.training_log, plot_path)
