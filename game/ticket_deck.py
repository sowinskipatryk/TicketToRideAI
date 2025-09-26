import random

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional
from collections import deque

from game.data_loader import load_tickets
from game.game_logger import logger


if TYPE_CHECKING:
    from game.core import Game


@dataclass(frozen=True)
class Ticket:
    city_from: str
    city_to: str
    points: int


class TicketDeck:
    def __init__(self, game: 'Game') -> None:
        self.game = game
        raw_tickets = load_tickets(self.game.version)
        self.tickets = [Ticket(city_from=t['from'], city_to=t['to'], points=int(t['points'])) for t in list(raw_tickets)]
        self.tickets_num = len(self.tickets)
        deck_list = self.tickets[:]
        random.shuffle(deck_list)
        self._ticket_deck = deque(deck_list)

    def get_tickets_num(self):
        return self.tickets_num

    def get_ticket_id(self, ticket: Ticket) -> int:
        return self.tickets.index(ticket)

    def insert(self, ticket: Ticket) -> None:
        self._ticket_deck.append(ticket)
        self.set_ticket_pile_num_adapter()

    def remove(self) -> Optional[Ticket]:
        if not self.is_empty():
            ticket = self._ticket_deck.popleft()
            self.set_ticket_pile_num_adapter()
            return ticket
        else:
            logger.info('Ticket deck is empty!')

    @property
    def tickets_left(self) -> int:
        return len(self._ticket_deck)

    def is_empty(self) -> bool:
        return self.tickets_left == 0

    def set_ticket_pile_num_adapter(self):
        self.game.adapter.set_ticket_pile_num(self.tickets_left)
