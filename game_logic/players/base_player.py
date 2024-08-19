import collections
from typing import List, Tuple, Dict, TYPE_CHECKING

from neat.nn import FeedForwardNetwork

from game_logic.boards.player_board import PlayerBoard
from game_logic.enums.decisions.actions import ActionDecision
from game_logic.enums.decisions.train_cards import TrainCardDecision
from game_logic.enums.player_colors import PlayerColor
from game_logic.game_logger import logger
from network.adapters.base_adapter import BaseAdapter

if TYPE_CHECKING:
    from game_logic.game import Game


class BasePlayer:
    PLAYER_COLORS = list(PlayerColor)

    def __init__(self, color_index: int, game_instance: 'Game', adapter: BaseAdapter,
                 network: FeedForwardNetwork = None) -> None:
        self.player_id = color_index
        self.game_instance = game_instance
        self.color = self.PLAYER_COLORS[color_index]
        self.tickets = {}
        self.hand = collections.defaultdict(int)
        self.score = 0
        self.trains_remaining = self.game_instance.config.TRAIN_FIGURES_NUM
        self.longest_path = False
        self.player_board = PlayerBoard(self)
        self.network = network
        self.adapter = adapter

        self.set_trains_num_adapter()

    def __str__(self) -> str:
        return str(self.color)

    def play_turn(self) -> bool:
        action_id = self.decide_action()
        action = ActionDecision(action_id)
        logger.info(f'play_turn: {action}')
        if action == ActionDecision.CLAIM_ROUTE:
            move_completed = self.claim_route()
        elif action == ActionDecision.DRAW_TICKETS:
            move_completed = self.draw_tickets(self.game_instance.config.TICKETS_DEALT_NUM,
                                               self.game_instance.config.TICKETS_TO_KEEP_NUM)
        elif action == ActionDecision.DRAW_CARDS:
            move_completed = self.draw_train_cards(self.game_instance.config.TRAIN_CARDS_DEALT_NUM)
        else:
            raise Exception("Invalid action")
        logger.info(f'move_completed: {move_completed}')
        return move_completed

    def add_ticket(self, ticket: tuple) -> None:
        logger.info(f'{self} picks ticket {ticket}')
        self.tickets[ticket] = False
        self.adapter.set_ticket_owner(self.player_id, ticket)

    def complete_ticket(self, ticket: tuple) -> None:
        if ticket not in self.tickets:
            raise ValueError(f'Ticket: {ticket} not found')
        self.tickets[ticket] = True
        self.adapter.set_ticket_completed(self.player_id, ticket)

    def score_tickets(self) -> None:
        for ticket, completed in self.tickets.items():
            ticket_value = self.get_ticket_value(ticket)
            if completed:
                self.add_points(ticket_value)
            else:
                self.subtract_points(ticket_value)

    @staticmethod
    def get_ticket_value(ticket: tuple) -> int:
        return ticket[2]

    def add_cards_to_hand(self, cards: str | List[str]) -> None:
        logger.info(f'add_cards_to_hand: {cards}')
        if not cards:
            return
        if not isinstance(cards, list):
            cards = [cards]
        for card in cards:
            self.hand[card] += 1
            self.set_cards_num_adapter(card)

    def remove_cards_from_hand(self, color: str, num_color: int) -> None:
        if self.hand[color] < num_color:
            raise ValueError('Not enough cards of the specified color')
        self.hand[color] -= num_color

        self.set_cards_num_adapter(color)

    def play_num_trains(self, num_trains: int) -> None:
        if num_trains > self.trains_remaining:
            raise ValueError('Not enough train figures')
        self.trains_remaining -= num_trains
        self.set_trains_num_adapter()

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
        return self.color

    def set_longest_path(self) -> None:
        self.longest_path = True

    def draw_tickets(self, num_tickets: int, min_keep: int) -> bool:
        tickets = self.game_instance.deal_tickets(num_tickets)

        if not tickets:
            return False

        kept, discarded = self.choose_tickets(min_keep, tickets)

        for index in range(len(tickets)):
            if index in kept:
                self.add_ticket(tickets[index])
                self.adapter.set_ticket_owner(self.player_id, tickets[index])
            else:
                logger.info(f'{self} discards ticket {tickets[index]}')
                self.game_instance.ticket_deck.insert(tickets[index])

        return True

    def draw_initial_train_cards(self, num_cards: int) -> bool:
        logger.info(f'draw_initial_train_cards')
        cards = [self.game_instance.deal_draw_pile_card() for _ in range(num_cards)]
        self.add_cards_to_hand(cards)
        return True

    def draw_train_cards(self, num_cards: int) -> bool:
        logger.info(f'draw_train_cards {num_cards}')
        for i in range(num_cards):
            train_card_decision_id = self.decide_train_card()
            train_card_decision = TrainCardDecision(train_card_decision_id)
            logger.info(f'train_card_decision: {train_card_decision}')

            if train_card_decision == TrainCardDecision.DRAW_PILE:
                train_card = self.game_instance.deal_draw_pile_card()
            elif train_card_decision in list(TrainCardDecision):
                train_card = self.game_instance.deal_face_up_card(train_card_decision_id)
                if train_card is None:
                    return False
                elif train_card == 'wild':
                    if self.game_instance.config.WILD_CARD_RESTRICTION:
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
        route_id = self.decide_route()
        city1, city2, route_data = self.game_instance.board.get_route_data(route_id)
        logger.info(f'chosen route: {city1, city2, route_data}')

        if self.color in route_data['claimed_by']:
            logger.info('you already claimed one path for that route!')
            return False

        if all(route_data['claimed_by']):
            logger.info('all paths for that route are claimed!')
            return False

        if any(route_data['claimed_by']) and self.game_instance.config.WILD_CARD_RESTRICTION and self.game_instance.players_num < 4:
            logger.info('one path for that route is already claimed! second path is restricted in this game configuration!')
            return False

        cards_color_id = self.decide_cards_color()
        cards_color = self.game_instance.config.TRAIN_COLORS[cards_color_id]

        logger.info(f'chosen card color: {cards_color} ({self.hand[cards_color]})')
        if cards_color not in route_data['edge_colors'] and 'grey' not in route_data['edge_colors']:
            logger.info(f"Color mismatch! Route has {', '.join(route_data['edge_colors'])} paths and chosen color is {cards_color}.")
            return False

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

        self.game_instance.board.claim_route(route_id, cards_color, self.color)
        self.remove_cards_from_hand(cards_color, color_cards_used_num)
        self.remove_cards_from_hand('wild', wild_cards_used_num)
        logger.debug(f'color_cards_used {color_cards_used_num}')
        logger.debug(f'cards_color {cards_color}')
        self.game_instance.train_card_manager.add_to_discard_pile([cards_color for _ in range(color_cards_used_num)])
        self.game_instance.train_card_manager.add_to_discard_pile(['wild' for _ in range(wild_cards_used_num)])
        self.play_num_trains(route_dist)
        self.add_points(self.game_instance.get_route_value(route_dist))
        self.player_board.add_edge(city1, city2, route_dist, cards_color)
        self.check_completed_tickets()

        self.adapter.set_trains_num(self.player_id, self.trains_remaining)
        self.adapter.set_route_owner(self.player_id, route_id)

        return True

    def check_completed_tickets(self) -> None:
        for ticket, completed in self.tickets.items():
            if not completed and self.player_board.is_ticket_completed(ticket):
                logger.info('TICKET_COMPLETED!')
                self.complete_ticket(ticket)

    def set_trains_num_adapter(self):
        self.adapter.set_trains_num(self.player_id, self.trains_remaining)

    def set_cards_num_adapter(self, card):
        if card == 'wild':
            self.adapter.set_wild_cards_num(self.player_id, self.hand[card])
        else:
            self.adapter.set_color_cards_num(self.player_id, card, self.hand[card])

    def choose_tickets(self, min_keep: int, tickets: List[tuple]) -> Tuple[List[int], List[int]]:
        raise NotImplementedError

    def decide_route(self) -> int:
        raise NotImplementedError

    def decide_cards_color(self) -> int:
        raise NotImplementedError

    def decide_train_card(self) -> int:
        raise NotImplementedError

    def decide_ticket(self) -> int:
        raise NotImplementedError

    def decide_action(self) -> int:
        raise NotImplementedError

    def decide_wild_cards(self) -> int:
        raise NotImplementedError
