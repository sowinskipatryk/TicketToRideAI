"""Logging configuration for the game."""
import logging
import os
from typing import Optional

# Try to load config, fall back to defaults if not available
try:
    from game.config_loader import load_config
    _config = load_config()
    _log_config = _config.get('logging', {})
    CONSOLE_LEVEL = _log_config.get('console_level', 'DEBUG')
    FILE_LEVEL = _log_config.get('file_level', 'ERROR')
    LOG_FILE = _log_config.get('log_file', 'game.log')
except (ImportError, Exception):
    # Default values if config loading fails
    CONSOLE_LEVEL = 'DEBUG'
    FILE_LEVEL = 'ERROR'
    LOG_FILE = 'game.log'

parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def get_log_level(level_name: str) -> int:
    """Convert string level name to logging constant.
    
    Args:
        level_name: Level name (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Logging level constant
    """
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    return level_map.get(level_name.upper(), logging.DEBUG)


def setup_logger(console_level: Optional[str] = None, 
                 file_level: Optional[str] = None,
                 log_file: Optional[str] = None) -> logging.Logger:
    """Setup and configure the game logger.
    
    Args:
        console_level: Console logging level (defaults to config)
        file_level: File logging level (defaults to config)
        log_file: Log file path (defaults to config)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Use provided values or fall back to config/defaults
    console_lvl = get_log_level(console_level or CONSOLE_LEVEL)
    file_lvl = get_log_level(file_level or FILE_LEVEL)
    log_path = log_file or LOG_FILE
    
    # Create file handler
    file_handler = logging.FileHandler(os.path.join(parent_directory, log_path))
    file_handler.setLevel(file_lvl)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_lvl)
    
    # Create formatter
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


# Create the logger instance
logger = setup_logger()
