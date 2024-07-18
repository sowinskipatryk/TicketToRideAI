import logging
import os

parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Create a logger object
logger = logging.getLogger('logger')

# Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
logger.setLevel(logging.DEBUG)

# Create a file handler and set its level
file_handler = logging.FileHandler(os.path.join(parent_directory, 'game.log'))
file_handler.setLevel(logging.ERROR)

# Create a console handler and set its level
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Create a formatter and set its format
formatter = logging.Formatter('[%(levelname)s] %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)
