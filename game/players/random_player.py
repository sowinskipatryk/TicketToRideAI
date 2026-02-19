import random
from typing import Tuple, List

from game.enums import ActionDecision, TrainCardDecision
from game.players.base_player import BasePlayer


class RandomPlayer(BasePlayer):
    def decide_tickets(self, min_kept: int, tickets: List[tuple]) -> Tuple[List[int], List[int]]:
        num_tickets = len(tickets)
        ticket_ids = list(range(num_tickets))
        random.shuffle(ticket_ids)
        min_kept = min(min_kept, num_tickets)
        num_kept = random.randint(min_kept, num_tickets)
        kept_ids = ticket_ids[:num_kept]
        discarded_ids = ticket_ids[num_kept:]
        return kept_ids, discarded_ids

    def decide_action(self) -> int:
        return random.randrange(0, len(list(ActionDecision)))

    def decide_wild_cards(self) -> int:
        return random.randint(0, self.hand.get('wild', 0))

    def decide_cards_color(self) -> int:
        return random.randrange(0, len(self.game.config.TRAIN_COLORS))

    def decide_train_card(self) -> int:
        return random.randrange(0, len(list(TrainCardDecision)))

    def decide_route(self) -> int:
        return random.randrange(0, self.game.board.get_route_links_num())
