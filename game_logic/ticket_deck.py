import random

from game_logic.utils import load_tickets
from game_logic.game_logger import logger


class TicketDeck:
    def __init__(self) -> None:
        self._ticket_deck = load_tickets()
        random.shuffle(self._ticket_deck)

    def insert(self, ticket: tuple) -> None:
        self._ticket_deck.append(ticket)

    def remove(self) -> tuple:
        if not self.is_empty:
            return self._ticket_deck.popleft()
        else:
            logger.info('Ticket deck is empty!')

    @property
    def is_empty(self) -> bool:
        return len(self._ticket_deck) == 0
