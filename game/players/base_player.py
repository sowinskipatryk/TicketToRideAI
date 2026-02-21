from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Tuple, Dict, TYPE_CHECKING

from game.enums import ActionDecision
from game.enums import TrainCardDecision, PlayerColor
from game.ticket_deck import Ticket
from game.game_logger import logger

if TYPE_CHECKING:
    from game.core import Game


class BasePlayer(ABC):
    """Base class for all player types.
    
    Defines the interface and common functionality for players.
    Subclasses must implement decision-making methods.
    """
    def __init__(self, color_index: int, game: 'Game') -> None:
        """Initialize a player.
        
        Args:
            color_index: Index determining player color
            game: Reference to the Game instance
            adapter: Adapter for state management (network or blank)
        """
        self.player_id = color_index
        self.game = game
        self.color = PlayerColor.from_index(color_index)
        self.tickets = {}
        self.hand = defaultdict(int)
        self.score = 0
        self.trains_remaining = self.game.config.NUM_TRAIN_FIGURES
        self.longest_path = False

    def __str__(self) -> str:
        return str(self.color)

    def play_turn(self) -> bool:
        """Execute a player's turn.
        
        Returns:
            True if the turn was completed successfully, False if the move was invalid
        """
        action_id = self.decide_action()
        action = ActionDecision(action_id)

        logger.info(f'play_turn: {action}')

        if action == ActionDecision.CLAIM_ROUTE:
            move_completed = self.claim_route()
        elif action == ActionDecision.DRAW_TICKETS:
            move_completed = self.draw_tickets(self.game.config.NUM_TICKETS_DEALT, self.game.config.MIN_TICKETS_KEPT)
        elif action == ActionDecision.DRAW_CARDS:
            move_completed = self.draw_train_cards(self.game.config.NUM_TRAIN_CARDS_DEALT)
        elif action == ActionDecision.SKIP:
            move_completed = True
        else:
            raise ValueError(f"Invalid action: {action} (action_id: {action_id})")
        if move_completed:
            logger.debug(f'move_completed')
        return move_completed

    def add_ticket(self, ticket: Ticket) -> None:
        self.tickets[ticket] = False

    def complete_ticket(self, ticket: Ticket) -> None:
        if ticket not in self.tickets:
            raise ValueError(f'Ticket: {ticket} not found')
        self.tickets[ticket] = True

    def score_tickets(self) -> None:
        for ticket, completed in self.tickets.items():
            if completed:
                self.add_points(ticket.points)
            else:
                self.subtract_points(ticket.points)

    def add_cards_to_hand(self, cards: str | List[str]) -> None:
        logger.debug(f'add_cards_to_hand: {cards}')
        if not cards:
            return
        if not isinstance(cards, list):
            cards = [cards]
        for card in cards:
            self.hand[card] += 1

    def remove_cards_from_hand(self, color: str, num_color: int) -> None:
        if self.hand[color] < num_color:
            raise ValueError('Not enough cards of the specified color')
        self.hand[color] -= num_color

    def play_num_trains(self, num_trains: int) -> None:
        if num_trains > self.trains_remaining:
            raise ValueError('Not enough train figures')
        self.trains_remaining -= num_trains

    def add_points(self, points_num: int) -> None:
        self.score += points_num

    def subtract_points(self, points_num: int) -> None:
        self.score -= points_num

    def get_trains_num(self) -> int:
        return self.trains_remaining

    def get_ticket_values(self) -> List[bool]:
        return list(self.tickets.values())

    def get_score(self) -> int:
        return self.score

    def get_hand(self) -> Dict[str, int]:
        return self.hand

    def get_color(self) -> str:
        return str(self.color)

    def set_longest_path(self) -> None:
        self.longest_path = True

    def draw_tickets(self, num_tickets: int, min_keep: int) -> bool:
        tickets = self.game.deal_tickets(num_tickets)

        if not tickets:
            logger.debug('no tickets')
            return False

        kept, discarded = self.decide_tickets(min_keep, tickets)

        for index in range(len(tickets)):
            if index in kept:
                logger.info(f'{self} picks ticket {tickets[index]}')
                self.add_ticket(tickets[index])
            else:
                logger.info(f'{self} discards ticket {tickets[index]}')
                self.game.ticket_deck.insert(tickets[index])

        return True

    def draw_initial_train_cards(self, num_cards: int) -> bool:
        logger.info(f'draw_initial_train_cards')
        cards = [self.game.deal_draw_pile_card() for _ in range(num_cards)]
        self.add_cards_to_hand(cards)
        return True

    def draw_train_cards(self, num_cards: int) -> bool:
        logger.info(f'draw_train_cards {num_cards}')
        for i in range(num_cards):
            train_card_decision_id = self.decide_train_card()
            train_card_decision = TrainCardDecision(train_card_decision_id)
            logger.info(f'train_card_decision: {train_card_decision}')

            if train_card_decision == TrainCardDecision.DRAW_PILE:
                train_card = self.game.deal_draw_pile_card()
            elif train_card_decision in list(TrainCardDecision):
                train_card = self.game.deal_face_up_card(train_card_decision_id)
                if train_card is None:
                    logger.info('no card in this position')
                    return False
                elif train_card == 'wild':
                    if self.game.config.WILD_CARD_RESTRICTION:
                        if i != 0:
                            return False
                        else:
                            self.add_cards_to_hand(train_card)
                            return True
                    else:
                        self.add_cards_to_hand(train_card)
            else:
                raise Exception('Invalid train card decision')

            if train_card is None:
                return False
            else:
                self.add_cards_to_hand(train_card)
        return True

    def claim_route(self) -> bool:
        route_link_id = self.decide_route()
        city1, city2, route_data = self.game.board.get_route_data(route_link_id)
        logger.info(f'chosen route: {city1, city2, route_data}')

        if not self.game.board.validate_route(self.color, route_data):
            return False

        cards_color = route_data['edge_color']
        if cards_color == 'grey':
            cards_color_id = self.decide_cards_color()
            cards_color = self.game.config.TRAIN_COLORS[cards_color_id]

        route_dist = route_data['weight']
        if route_dist > self.trains_remaining:
            logger.info('too few train figures!')
            return False

        wild_cards_num_decision = self.decide_wild_cards()
        wild_cards_used_num = min(wild_cards_num_decision, self.hand['wild'])
        color_cards_used_num = route_dist - wild_cards_used_num

        logger.info(f'wild cards num: {wild_cards_used_num}')

        if self.hand[cards_color] + wild_cards_used_num < route_dist:
            return False

        self.game.board.claim_route(route_link_id, self.color)
        self.remove_cards_from_hand(cards_color, color_cards_used_num)
        self.remove_cards_from_hand('wild', wild_cards_used_num)
        logger.debug(f'color_cards_used {color_cards_used_num}')
        logger.debug(f'cards_color {cards_color}')
        self.game.train_card_manager.add_to_discard_pile([cards_color for _ in range(color_cards_used_num)])
        self.game.train_card_manager.add_to_discard_pile(['wild' for _ in range(wild_cards_used_num)])
        self.play_num_trains(route_dist)
        self.add_points(self.game.get_route_value(route_dist))
        self.check_completed_tickets()

        return True

    def check_completed_tickets(self) -> None:
        for ticket, completed in self.tickets.items():
            if not completed and self.game.board.is_ticket_completed(self.color, ticket):
                logger.info('TICKET_COMPLETED!')
                self.complete_ticket(ticket)

    @abstractmethod
    def decide_tickets(self, min_keep: int, tickets: List[Ticket]) -> Tuple[List[int], List[int]]:
        pass

    @abstractmethod
    def decide_route(self) -> int:
        pass

    @abstractmethod
    def decide_cards_color(self) -> int:
        pass

    @abstractmethod
    def decide_train_card(self) -> int:
        pass

    @abstractmethod
    def decide_action(self) -> int:
        pass

    @abstractmethod
    def decide_wild_cards(self) -> int:
        pass
