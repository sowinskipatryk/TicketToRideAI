"""Configuration loader for game settings."""
import os
import yaml
from typing import Dict, Any


_DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')


def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary containing configuration values
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
    """
    if config_path is None:
        config_path = _DEFAULT_CONFIG_PATH
    if not os.path.exists(config_path):
        # Return default configuration if file doesn't exist
        return get_default_config()
    
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    
    return config if config else get_default_config()


def get_default_config() -> Dict[str, Any]:
    """Get default configuration values.
    
    Returns:
        Dictionary with default configuration
    """
    return {
        'neat': {
            'players_num': 4,
            'game_version': 'USA',
            'num_generations': 20,
            'max_moves_per_game': 1000,
            'config_filename': 'neuroevolution/neat_config.txt',
            'genome_filename': 'checkpoints/neat/best_genome.pkl'
        },
        'logging': {
            'console_level': 'DEBUG',
            'file_level': 'ERROR',
            'log_file': 'game.log'
        },
        'game': {
            'default_version': 'USA',
            'min_players': 2,
            'max_players': 5
        }
    }

