# TicketToRideAI

A Python implementation of the Ticket to Ride board game with multiple AI players, including AlphaZero (self-play + MCTS), Information Set MCTS, and NEAT neuroevolution.

## Features

- **Multiple AI Paradigms**: AlphaZero (policy+value network + self-play), IS-MCTS (tree search with determinization), NEAT (neuroevolution)
- **Imperfect Information Handling**: Determinization-based approach for hidden opponent hands
- **Multiple Game Versions**: USA and Europe variants
- **10+ Player Types**: Human, Random, Greedy, MCTS, AlphaZero, NEAT, TicketFocused, CardHoarder, Blocker
- **GUI**: PyQt6-based graphical interface
- **Training Infrastructure**: Self-play, replay buffer, checkpointing, baseline evaluation
- **Telemetry**: SQLite-based game data collection for analysis
- **Batch Simulation**: Run and compare N games in parallel

## Installation

### Prerequisites

- Python 3.8+
- pip
- CUDA-capable GPU (recommended for AlphaZero training)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd TicketToRideAI
```

2. Create a virtual environment:
```bash
python -m venv .venv
```

3. Activate the virtual environment:
   - Windows: `.venv\Scripts\activate`
   - Linux/Mac: `source .venv/bin/activate`

4. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Playing a Game (GUI)

```bash
python scripts/play_game_gui.py
```

### Playing a Game (CLI)

```bash
python scripts/play_game.py
```

Edit `scripts/play_game.py` to configure players and version:
```python
players = ['AlphaZero', 'MCTS']
version = 'USA'
```

### Training AlphaZero

```bash
python scripts/train_alphazero.py \
  --iterations 100 \
  --games 50 \
  --mcts-iters 200 \
  --batch-size 256 \
  --hidden-size 256 \
  --res-blocks 4 \
  --lr 0.001 \
  --checkpoint-dir checkpoints/alphazero
```

Resume from checkpoint:
```bash
python scripts/train_alphazero.py --resume --checkpoint-dir checkpoints/alphazero
```

Training automatically uses GPU if available, CPU otherwise. Self-play runs on CPU (batch=1 GPU inference has too much transfer overhead); network training runs on CUDA.

### Training NEAT

```bash
python scripts/train_neat.py
```

Configure in `neuroevolution/neat_config.txt` and `config.yaml`.

### Evaluating NEAT

```bash
python scripts/eval_neat.py
```

Tests best genome against baseline opponents (Greedy, TicketFocused, etc.), tracking win rate and average score difference.

### Batch Simulation

```bash
python scripts/batch_simulate.py \
  --games 100 \
  --players AlphaZero MCTS \
  --version USA \
  --parallel
```

### Configuration

Edit `config.yaml` to adjust:
- `neat.players_num`, `neat.num_generations`, `neat.game_version`
- `logging.console_level`, `logging.file_level`, `logging.log_file`
- `game.default_version`

## Architecture

### Core Components

- **Game Engine** (`game/core.py`): Full Ticket to Ride rules for 2–5 players
- **Simulation Engine** (`game/sim_game.py`): Lightweight, fast clone-friendly state for MCTS/self-play (no GUI/player dependencies)
- **Game Controller** (`game/game_controller.py`): Wraps game with event system and state snapshots
- **Player Factory** (`game/player_factory.py`): Instantiates any player type by name

### AI Systems

#### AlphaZero (`alphazero/`)

Self-play training with a dual-head residual network guided by MCTS.

| File | Purpose |
|------|---------|
| `network.py` | Dual-head ResNet: policy head (907 action logits) + value head (scalar in [-1, 1]) |
| `encoding.py` | 389-feature state vector + 907-action space (USA version hardcoded) |
| `az_mcts.py` | PUCT-based MCTS with NN evaluation, Dirichlet noise at root, determinization |
| `self_play.py` | Generates training samples with temperature annealing (τ=1.0 → 0.1 at move 15) |
| `replay_buffer.py` | Circular buffer (100k samples) for experience replay |
| `trainer.py` | Full training loop: self-play → buffer → minibatch training → checkpoint |

#### Information Set MCTS (`mcts/`)

Classic MCTS adapted for imperfect information via determinization.

| File | Purpose |
|------|---------|
| `is_mcts.py` | IS-MCTS with UCB1 selection, configurable iterations (default 200) |
| `determinize.py` | Samples plausible opponent hands from known information |
| `rollout_policies.py` | `random_rollout` and `heuristic_rollout` (ticket-aware greedy) |

#### NEAT (`neuroevolution/`)

Neuroevolution with 1-step greedy lookahead evaluation.

| File | Purpose |
|------|---------|
| `manager.py` | Training orchestration, calibration vs baselines, checkpoint saving |
| `neat_config.txt` | NEAT algorithm hyperparameters |

### State Encoding

Both NEAT and AlphaZero use the same 389-feature vector:
- Player hand (card counts per color)
- Opponent hand sizes (not contents — imperfect info)
- Face-up cards
- Route ownership (all routes, all players)
- Destination tickets (own, completion status)
- Remaining train count
- Game phase

**Note**: Encoding is hardcoded for USA version. Switching to Europe requires retraining.

### Player Types

| Type | Description |
|------|-------------|
| `Human` | Interactive via CLI or GUI |
| `Random` | Uniformly random legal moves |
| `Greedy` | Maximizes immediate route points |
| `TicketFocused` | Prioritizes ticket completion |
| `CardHoarder` | Accumulates cards before claiming |
| `Blocker` | Targets opponent-needed routes |
| `MCTS` | IS-MCTS with heuristic rollouts (400 iters) |
| `AlphaZero` | MCTS guided by trained ResNet |
| `NEAT` | Evolved neural network, 1-step lookahead |

## Project Structure

```
TicketToRideAI/
├── alphazero/               # AlphaZero training system
│   ├── network.py           # Dual-head residual network
│   ├── encoding.py          # State/action encoding (USA, 389 features, 907 actions)
│   ├── az_mcts.py           # PUCT MCTS with NN evaluation
│   ├── self_play.py         # Self-play game generation
│   ├── replay_buffer.py     # Circular experience replay buffer
│   └── trainer.py           # Training loop orchestrator
├── mcts/                    # Information Set MCTS
│   ├── is_mcts.py           # IS-MCTS algorithm
│   ├── determinize.py       # Hidden info sampling
│   └── rollout_policies.py  # Random and heuristic rollouts
├── game/                    # Core game engine
│   ├── core.py              # Main game class
│   ├── sim_game.py          # Lightweight simulation state
│   ├── game_controller.py   # State/event wrapper
│   ├── game_board.py        # NetworkX route graph
│   ├── player_factory.py    # Player instantiation
│   ├── players/             # All player implementations
│   └── data/                # YAML game data (USA/, Europe/)
├── neuroevolution/          # NEAT training
│   ├── manager.py           # Training orchestration
│   └── neat_config.txt      # NEAT hyperparameters
├── gui/                     # PyQt6 graphical interface
├── cli/                     # Terminal human player interface
├── telemetry/               # SQLite telemetry collection
├── scripts/                 # Entry point scripts
│   ├── train_alphazero.py   # AlphaZero training
│   ├── train_neat.py        # NEAT training
│   ├── eval_neat.py         # NEAT evaluation
│   ├── batch_simulate.py    # Batch game runner
│   ├── play_game.py         # CLI game
│   └── play_game_gui.py     # GUI game
├── checkpoints/             # Saved checkpoints
│   ├── alphazero/           # AlphaZero model + training_log.json
│   └── neat/                # NEAT best_genome.pkl
├── config.yaml              # Main configuration
└── requirements.txt         # Python dependencies
```

## Game Rules Summary

1. **Objective**: Connect cities by claiming routes to complete destination tickets
2. **Turns**: Each turn, a player may:
   - Claim a route (spend matching train cards)
   - Draw train cards (face-up or blind from deck)
   - Draw destination tickets
3. **Scoring**: Route length points + completed ticket bonuses + longest route bonus
4. **Game End**: Triggered when a player reaches ≤3 remaining train pieces; all players get one final turn

## Development

### Adding a New Player

1. Create a class in `game/players/` inheriting from `BasePlayer`
2. Implement abstract methods: `decide_action()`, `decide_route()`, `decide_cards_color()`, `decide_train_card()`, `decide_tickets()`, `decide_wild_cards()`
3. Register in `game/player_factory.py` and `game/enums.py`

### Adding a New Game Version

1. Add data files under `game/data/<version>/`: `cities.yaml`, `routes.yaml`, `tickets.yaml`, `city_coordinates.yaml`
2. Create a config class in `game/config.py` inheriting from `BaseConfig`
3. Register in `GameVersion` enum and `ConfigFactory`
4. Re-encode if using with neural networks (encoding is version-specific)

## Troubleshooting

**AlphaZero checkpoint not found**: Train first with `scripts/train_alphazero.py`, or use `MCTS`/`Random` players.

**NEAT genome not found**: Train first with `scripts/train_neat.py`. Genome saved to `checkpoints/neat/best_genome.pkl`.

**GUI not starting**: Ensure PyQt6 is installed: `pip install PyQt6`.

**Slow AlphaZero training**: Verify CUDA is available (`torch.cuda.is_available()`). Self-play intentionally runs on CPU.

## License

[Add your license here]

## Acknowledgments

- Based on Ticket to Ride by Days of Wonder
- [NEAT-Python](https://neat-python.readthedocs.io/) for neuroevolution
- [PyTorch](https://pytorch.org/) for deep learning
- [PyQt6](https://www.riverbankcomputing.com/software/pyqt/) for GUI
