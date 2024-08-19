import random

from typing import TYPE_CHECKING

from game_logic.game_logger import logger
from game_logic.utils import load_tickets


if TYPE_CHECKING:
    from game_logic.game import Game


class TicketDeck:
    def __init__(self, game_instance: 'Game') -> None:
        self.game_instance = game_instance
        self._ticket_deck = load_tickets(self.game_instance.version)
        self.tickets = list(self._ticket_deck)
        self.tickets_num = len(self._ticket_deck)
        random.shuffle(self._ticket_deck)

        self.set_ticket_pile_num_adapter()

    def get_tickets_num(self):
        return self.tickets_num

    def get_ticket_id(self, ticket):
        return self.tickets.index(ticket)

    def insert(self, ticket: tuple) -> None:
        self._ticket_deck.append(ticket)
        self.set_ticket_pile_num_adapter()

    def remove(self) -> tuple:
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
        self.game_instance.adapter.set_ticket_pile_num(self.tickets_left)
