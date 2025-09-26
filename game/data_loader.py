import collections
import os
import yaml
from typing import List

current_dir = os.path.dirname(__file__)


def load_cities(version: str) -> List[str]:
    file_path = os.path.join(current_dir, 'data', version, 'cities.yaml')
    with open(file_path, 'r', encoding='utf-8') as file:
        cities = yaml.safe_load(file)
    return cities


def load_routes(version: str) -> List[List[str]]:
    file_path = os.path.join(current_dir, 'data', version, 'routes.yaml')
    routes = []
    with open(file_path, 'r', encoding='utf-8') as file:
        routes = yaml.safe_load(file)
    return routes


def load_tickets(version: str) -> collections.deque:
    file_path = os.path.join(current_dir, 'data', version, 'tickets.yaml')
    with open(file_path, 'r', encoding='utf-8') as file:
        tickets = yaml.safe_load(file)
    return collections.deque(tickets)
