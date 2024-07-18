import random
from typing import Tuple, List

from game_logic.players.base_player import BasePlayer


class RandomPlayer(BasePlayer):
    def choose_tickets(self, min_keep: int, tickets_num: int) -> Tuple[List[int], List[int]]:
        ticket_ids = list(range(tickets_num))
        random.shuffle(ticket_ids)
        min_keep = min(min_keep, tickets_num)
        keep_num = random.randint(min_keep, tickets_num)
        return ticket_ids[:keep_num], ticket_ids[keep_num:]

    def decide_action(self) -> int:
        return random.randint(0, 2)

    def decide_wild_cards(self) -> int:
        return random.randint(0, 14)

    def decide_cards_color(self) -> int:
        return random.randint(0, 7)

    def decide_train_card(self) -> int:
        return random.randint(0, 5)

    def decide_route(self) -> int:
        return random.randint(0, 99)
