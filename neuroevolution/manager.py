"""NEAT network training manager."""
import neat
import os
import pickle

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
        networks = []
        for j in range(PLAYERS_NUM):
            genome_id, genome = genomes[i + j]
            networks.append(neat.nn.FeedForwardNetwork.create(genome, config))

        game = Game(player_types=['NEAT'] * PLAYERS_NUM, version=GAME_VERSION, networks=networks)
        stats = game.play(max_moves=MAX_MOVES_PER_GAME)

        scores = stats['score']
        for j in range(PLAYERS_NUM):
            genome_id, genome = genomes[i + j]
            my_score = scores[j]
            opp_score = max(scores[k] for k in range(PLAYERS_NUM) if k != j)
            genome.fitness = my_score - opp_score
            print(f"Game {i // PLAYERS_NUM}, Player {j}, Genome {genome_id}, "
                  f"Score {my_score}, Fitness: {genome.fitness:+d}")


def save_genome(genome):
    with open(GENOME_FILENAME, 'wb') as file:
        pickle.dump(genome, file)


def run_neat():
    config_path = os.path.join(os.path.dirname(__file__), CONFIG_FILENAME)
    cnf = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                      neat.DefaultSpeciesSet, neat.DefaultStagnation,
                      config_path)

    # population = neat.Checkpointer.restore_checkpoint('neat-checkpoint-3')
    population = neat.Population(cnf)

    population.add_reporter(neat.StdOutReporter(show_species_detail=True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    population.add_reporter(neat.Checkpointer(generation_interval=1))

    winner = population.run(eval_genomes, NUM_GENERATIONS)
    save_genome(winner)
