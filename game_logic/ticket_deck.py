import random

from utils import load_tickets


class TicketDeck:
    def __init__(self):
        self._ticket_deck = load_tickets()
        self.shuffle()

    def insert_ticket(self, ticket):
        self._ticket_deck.append(ticket)

    def draw_ticket(self):
        if not self.is_empty:
            return self._ticket_deck.popleft()

    @property
    def is_empty(self):
        return len(self._ticket_deck) == 0

    def shuffle(self):
        random.shuffle(self._ticket_deck)
