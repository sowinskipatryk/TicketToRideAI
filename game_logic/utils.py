import collections
import os
from typing import List

current_dir = os.path.dirname(__file__)


def load_cities() -> List[str]:
    file_path = os.path.join(current_dir, 'data', 'cities.txt')
    with open(file_path, 'r') as file:
        lines = file.readlines()

    cities = []
    for line in lines:
        city = line.strip()
        cities.append(city)
    return cities


def load_routes() -> List[List[str]]:
    file_path = os.path.join(current_dir, 'data', 'routes.txt')
    with open(file_path, 'r') as file:
        lines = file.readlines()

    routes = []
    for line in lines:
        sline = line.strip()
        route = sline.split(', ')
        routes.append(route)
    return routes


def load_tickets() -> collections.deque:
    file_path = os.path.join(current_dir, 'data', 'tickets.txt')
    with open(file_path, 'r') as file:
        lines = file.readlines()

    tickets = collections.deque()
    for line in lines:
        parts = line.strip().split(', ')
        tickets.append((parts[0], parts[1], int(parts[2])))
    return tickets
