# TicketToRideAI

A Python implementation of the Ticket to Ride board game with AI players powered by NEAT (NeuroEvolution of Augmented Topologies) neural networks.

## Features

- **Multiple Game Versions**: Support for USA, Europe, and Nordic variants
- **AI Players**: NEAT-based neural network players that learn through evolution
- **Player Types**: Human, Random, and NEAT AI players
- **Extensible Architecture**: Easy to add new player types or game versions
- **GUI**: PyQt6-based graphical interface
- **State Management**: Serializable game state for debugging, replay, and AI cloning
- **Event System**: Complete event tracking for replay and debugging

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd TicketToRideAI
```

2. Create a virtual environment (recommended):
```bash
python -m venv .venv
```

3. Activate the virtual environment:
   - On Windows:
     ```bash
     .venv\Scripts\activate
     ```
   - On Linux/Mac:
     ```bash
     source .venv/bin/activate
     ```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Playing a Game (GUI)

Launch the graphical interface:

```bash
python scripts/play_game_gui.py
```

The GUI provides:
- Interactive game board
- Visual card and ticket display
- Player information panel
- Game log
- Action buttons

### Playing a Game (CLI)

Run a game with the command-line script:

```bash
python scripts/play_game.py
```

You can modify `scripts/play_game.py` to customize:
- **Players**: Choose from `'Human'`, `'NEAT'`, or `'Random'` (2-5 players)
- **Game Version**: Choose from `'USA'`, `'Europe'`, or `'Nordic'`

Example:
```python
players = ['Human', 'NEAT', 'Random', 'Random']
version = 'USA'
```

### Training NEAT Networks

Train a NEAT neural network to play the game:

```bash
python scripts/train_network.py
```

Training parameters can be configured in:
- `network/neat_config.txt` - NEAT algorithm configuration
- `network/manager.py` - Training parameters (players, generations, etc.)
- `config.yaml` - General configuration

After training, the best genome is saved as `best_genome.pkl` in the project root.

### Configuration

#### NEAT Training Configuration

Edit `network/manager.py` or `config.yaml` to adjust:
- `PLAYERS_NUM`: Number of players per game (default: 4)
- `GAME_VERSION`: Game version to train on (default: 'USA')
- `NUM_GENERATIONS`: Number of generations to evolve (default: 20)

#### Logging Configuration

Logging levels can be configured in `game/game_logger.py` or `config.yaml`:
- Console logging: Set `console_level` (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- File logging: Set `file_level`
- Log file: Set `log_file` path

## Architecture

The project uses a modular architecture with clear separation of concerns:

### Core Components

- **Game Logic** (`game/core.py`): Core game mechanics (unchanged)
- **Game Controller** (`game/game_controller.py`): Wraps game with state management and events
- **Game State** (`game/game_state.py`): Serializable state classes
- **Events** (`game/events.py`): Event system for replay and UI updates

### GUI Components

- **Main Window** (`gui/main_window.py`): Main application window
- **Game Board Widget** (`gui/game_board_widget.py`): Interactive board visualization
- **Player Info Widget** (`gui/player_info_widget.py`): Player information display
- **Card Widget** (`gui/card_widget.py`): Card and ticket display
- **Action Panel** (`gui/action_panel.py`): Action buttons
- **Game Controller GUI** (`gui/game_controller_gui.py`): Connects game logic to GUI

### Key Features

#### State Management
- **Snapshots**: Save game state at any point
- **Cloning**: Clone state for AI exploration
- **Serialization**: Save/load games as JSON

#### Event System
- **Event Tracking**: All game events are tracked
- **Replay**: Replay games from event history
- **Observer Pattern**: UI subscribes to events

#### Debugging
- **State Inspection**: Inspect state at any point
- **Event History**: See all events in order
- **State Comparison**: Compare states for debugging

## Project Structure

```
TicketToRideAI/
├── game/                    # Core game logic
│   ├── core.py             # Main game class
│   ├── game_controller.py  # Controller with state/events
│   ├── game_state.py       # Serializable state classes
│   ├── events.py           # Event system
│   ├── config.py           # Game configuration
│   ├── game_board.py       # Board and route management
│   ├── players/            # Player implementations
│   │   ├── base_player.py  # Base player class
│   │   ├── human_player.py # Human player
│   │   ├── neat_player.py  # NEAT AI player
│   │   └── random_player.py # Random player
│   └── data/               # Game data (YAML files)
│       ├── USA/
│       ├── Europe/
│       └── Nordic/
├── gui/                     # GUI components
│   ├── main_window.py      # Main window
│   ├── game_board_widget.py # Board widget
│   ├── player_info_widget.py # Player info
│   ├── card_widget.py      # Card display
│   ├── action_panel.py     # Action buttons
│   ├── game_controller_gui.py # GUI controller
│   └── new_game_dialog.py  # Game setup
├── network/                 # NEAT network integration
│   ├── manager.py          # Training manager
│   ├── decisions.py        # Network decision constants
│   ├── neat_config.txt     # NEAT algorithm config
│   └── adapters/           # State adapters for networks
├── scripts/                # Entry point scripts
│   ├── play_game.py        # CLI play game
│   ├── play_game_gui.py    # GUI play game
│   └── train_network.py    # Train NEAT network
├── tests/                  # Unit tests
├── config.yaml            # Configuration file
└── requirements.txt        # Python dependencies
```

## Game Rules

This implementation follows the standard Ticket to Ride rules:

1. **Objective**: Connect cities by claiming routes to complete destination tickets
2. **Turns**: Each turn, players can either:
   - Claim a route (requires matching train cards)
   - Draw train cards (from face-up cards or draw pile)
   - Draw destination tickets
3. **Scoring**: Points are awarded for:
   - Claimed routes (based on length)
   - Completed destination tickets
   - Longest continuous route (bonus)
4. **Game End**: Triggered when a player has 3 or fewer train pieces remaining

## Player Types

### Human Player
Interactive player that prompts for decisions via command line or GUI.

### Random Player
Makes random valid moves. Useful for testing and as a baseline.

### NEAT Player
AI player using evolved neural networks. Requires a trained genome file (`best_genome.pkl`).

## Development

### Adding a New Player Type

1. Create a new class in `game/players/` inheriting from `BasePlayer`
2. Implement all abstract methods:
   - `decide_action()`
   - `decide_route()`
   - `decide_cards_color()`
   - `decide_train_card()`
   - `decide_tickets()`
   - `decide_wild_cards()`
3. Register in `game/player_factory.py`

### Adding a New Game Version

1. Create a new config class in `game/config.py` inheriting from `BaseConfig`
2. Add game data files in `game/data/<version>/`:
   - `cities.yaml`
   - `routes.yaml`
   - `tickets.yaml`
3. Register in `GameVersion` enum and `ConfigFactory`

### Using State Management

```python
from game.game_controller import GameController

controller = GameController(player_types, version)

# Get current state
snapshot = controller.get_state_snapshot()

# Clone for AI exploration
clone = snapshot.clone()

# Save snapshot
controller.save_snapshot()

# Access event history
events = controller.event_bus.get_event_history()
```

### Using Events

```python
from game.events import EventType

def on_route_claimed(event):
    print(f"Route claimed: {event.data}")

controller.event_bus.subscribe(EventType.ROUTE_CLAIMED, on_route_claimed)
```

## Testing

Run tests (when available):
```bash
python -m pytest tests/
```

## Troubleshooting

### "The genome file was not found!"
- Train a network first using `python scripts/train_network.py`
- Or use `'Random'` or `'Human'` players instead of `'NEAT'`

### GUI not starting
- Ensure PyQt6 is installed: `pip install PyQt6`
- Check Python version (3.8+ required)

### Game hangs or crashes
- Check that you have valid player configurations (2-5 players)
- Ensure game data files are present in `game/data/`
- Check logs in `game.log` for error details

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

## Acknowledgments

- Based on the Ticket to Ride board game by Days of Wonder
- Uses NEAT-Python library for neural network evolution
- GUI built with PyQt6
