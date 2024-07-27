from typing import List

from game_logic.boards.game_board import GameBoard
from game_logic.ticket_deck import TicketDeck
from game_logic.train_card_manager import TrainCardManager
from game_logic.game_config import GameConfigAmerica


class GameManager:
    def __init__(self, players_num):
        self.players_num = players_num
        self.board = GameBoard()
        self.ticket_deck = TicketDeck()
        self.train_card_manager = TrainCardManager()

    def deal_face_up_card(self, card_id):
        return self.train_card_manager.pick_face_up_card(card_id)

    def deal_draw_pile_card(self):
        return self.train_card_manager.pick_draw_pile_card()

    def deal_tickets(self, num_tickets: int) -> List[tuple]:
        dealt_tickets = []
        while len(dealt_tickets) < num_tickets:
            ticket = self.ticket_deck.remove()
            if ticket is None:
                break
            dealt_tickets.append(ticket)
        return dealt_tickets

    @staticmethod
    def get_route_value(route_length: int) -> int:
        return GameConfigAmerica.ROUTE_VALUES[route_length]
