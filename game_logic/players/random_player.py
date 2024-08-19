import random
from typing import Tuple, List

from game_logic.enums.decisions.actions import ActionDecision
from game_logic.enums.decisions.train_cards import TrainCardDecision
from game_logic.players.base_player import BasePlayer


class RandomPlayer(BasePlayer):
    def choose_tickets(self, min_keep: int, tickets: List[tuple]) -> Tuple[List[int], List[int]]:
        tickets_num = len(tickets)
        ticket_ids = list(range(tickets_num))
        random.shuffle(ticket_ids)
        min_keep = min(min_keep, tickets_num)
        keep_num = random.randint(min_keep, tickets_num)
        keep_ids = ticket_ids[:keep_num]
        discard_ids = ticket_ids[keep_num:]
        return keep_ids, discard_ids

    def decide_action(self) -> int:
        return random.randrange(0, len(list(ActionDecision)))

    def decide_wild_cards(self) -> int:
        return random.randint(0, self.game_instance.config.WILD_CARDS_NUM)

    def decide_cards_color(self) -> int:
        return random.randrange(0, len(self.game_instance.config.TRAIN_COLORS))

    def decide_train_card(self) -> int:
        return random.randrange(0, len(list(TrainCardDecision)))

    def decide_route(self) -> int:
        return random.randrange(0, self.game_instance.board.get_links_num())
