import collections
import os
from typing import List

current_dir = os.path.dirname(__file__)


def load_cities(version: str) -> List[str]:
    file_path = os.path.join(current_dir, 'data', version, 'cities.txt')
    cities = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            city = line.strip()
            cities.append(city)
    return cities


def load_routes(version: str) -> List[List[str]]:
    file_path = os.path.join(current_dir, 'data', version, 'routes.txt')
    routes = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            sline = line.strip()
            route = sline.split(', ')
            routes.append(route)
    return routes


def load_tickets(version: str) -> collections.deque:
    file_path = os.path.join(current_dir, 'data', version, 'tickets.txt')
    tickets = collections.deque()
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(', ')
            tickets.append((parts[0], parts[1], int(parts[2])))
    return tickets
