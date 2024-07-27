import collections
from typing import List, Tuple

from game_logic.game_config import GameConfigAmerica
from game_logic.enums.decisions.actions import ActionDecision
from game_logic.enums.decisions.train_cards import TrainCardDecision
from game_logic.enums.player_colors import PlayerColor
from game_logic.game_logger import logger
from game_logic.boards.player_board import PlayerBoard


class BasePlayer:
    PLAYER_COLORS = list(PlayerColor)

    def __init__(self, color_index: int) -> None:
        self.color = self.PLAYER_COLORS[color_index]
        self.tickets = {}
        self.hand = collections.defaultdict(int)
        self.score = 0
        self.trains_remaining = GameConfigAmerica.TRAIN_FIGURES_NUM
        self.longest_path = False
        self.player_board = PlayerBoard(self)

    def __str__(self) -> str:
        return str(self.color)

    def play_turn(self, game_manager):
        action_id = self.decide_action()
        action = ActionDecision(action_id)
        logger.info(f'play_turn: {action}')
        if action == ActionDecision.CLAIM_ROUTE:
            move_completed = self.claim_route(game_manager)
            # game_manager.board.draw_possession_graph()
        elif action == ActionDecision.DRAW_TICKETS:
            move_completed = self.draw_tickets(game_manager, GameConfigAmerica.TICKETS_DEALT_NUM, GameConfigAmerica.TICKETS_TO_KEEP_NUM)
        elif action == ActionDecision.DRAW_CARDS:
            move_completed = self.draw_train_cards(game_manager, GameConfigAmerica.TRAIN_CARDS_DEALT_NUM)
        else:
            raise Exception("Invalid action")
        logger.info(f'move_completed: {move_completed}')
        return move_completed

    def add_ticket(self, ticket):
        logger.info(f'{self} picks ticket {ticket}')
        self.tickets[ticket] = False

    def complete_ticket(self, ticket):
        assert ticket in self.tickets
        self.tickets[ticket] = True

    def score_tickets(self):
        for ticket, completed in self.tickets.items():
            ticket_value = self.get_ticket_value(ticket)
            if completed:
                self.add_points(ticket_value)
            else:
                self.subtract_points(ticket_value)

    @staticmethod
    def get_ticket_value(ticket):
        return ticket[2]

    def add_cards_to_hand(self, cards):
        logger.info(f'add_cards_to_hand: {cards}')
        if not cards:
            return
        if not isinstance(cards, list):
            cards = [cards]
        for card in cards:
            self.hand[card] += 1

    def remove_cards_from_hand(self, color, num_color):
        assert self.hand[color] >= num_color
        self.hand[color] -= num_color

    def play_num_trains(self, num_trains):
        assert num_trains <= self.trains_remaining
        self.trains_remaining -= num_trains

    def add_points(self, points_num):
        self.score += points_num

    def subtract_points(self, points_num):
        self.score -= points_num

    def get_trains_num(self):
        return self.trains_remaining

    def get_ticket_values(self):
        return self.tickets.values()

    def get_score(self):
        return self.score

    def get_hand(self):
        return self.hand

    def get_color(self):
        return self.color

    def set_longest_path(self):
        self.longest_path = True

    def draw_tickets(self, game_manager, num_tickets, min_keep):
        tickets = game_manager.deal_tickets(num_tickets)

        if not tickets:
            return False

        tickets_num = len(tickets)
        kept, discarded = self.choose_tickets(min_keep, tickets_num)

        for index in range(len(tickets)):
            if index in kept:
                self.add_ticket(tickets[index])
            else:
                logger.info(f'{self} discards ticket {tickets[index]}')
                game_manager.ticket_deck.insert(tickets[index])

    def draw_initial_train_cards(self, game_manager, num_cards):
        logger.info(f'draw_initial_train_cards')
        cards = [game_manager.deal_draw_pile_card() for _ in range(num_cards)]
        self.add_cards_to_hand(cards)

    def draw_train_cards(self, game_manager, num_cards):
        logger.info(f'draw_train_cards {num_cards}')
        for i in range(num_cards):
            train_card_decision_id = self.decide_train_card()
            train_card_decision = TrainCardDecision(train_card_decision_id)
            logger.info(f'train_card_decision: {train_card_decision}')

            if train_card_decision == TrainCardDecision.DRAW_PILE:
                train_card = game_manager.deal_draw_pile_card()
            elif train_card_decision in list(TrainCardDecision):
                card_id = self.decide_train_card()
                train_card = game_manager.deal_face_up_card(card_id)
                if train_card is None:
                    return False
                elif train_card == 'wild':
                    if GameConfigAmerica.WILD_CARD_RESTRICTION:
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

    def claim_route(self, game_manager):
        route_id = self.decide_route()
        city1, city2, route_data = game_manager.board.get_route_data(route_id)
        logger.info(f'chosen route: {city1, city2, route_data}')

        if self.color in route_data['claimed_by']:
            logger.info('you already claimed one path for that route!')
            return False

        if all(route_data['claimed_by']):
            logger.info('all paths for that route are claimed!')
            return False

        if any(route_data['claimed_by']) and GameConfigAmerica.WILD_CARD_RESTRICTION and game_manager.players_num < 4:
            logger.info('one path for that route is already claimed! second path is restricted in this game configuration!')
            return False

        cards_color_id = self.decide_cards_color()
        cards_color = GameConfigAmerica.TRAIN_COLORS[cards_color_id]

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

        game_manager.board.claim_route(route_id, cards_color, self.color)
        self.remove_cards_from_hand(cards_color, color_cards_used_num)
        self.remove_cards_from_hand('wild', wild_cards_used_num)
        logger.debug(f'color_cards_used {color_cards_used_num}')
        logger.debug(f'cards_color {cards_color}')
        game_manager.train_card_manager.add_to_discard_pile([cards_color for _ in range(color_cards_used_num)])
        game_manager.train_card_manager.add_to_discard_pile(['wild' for _ in range(wild_cards_used_num)])
        self.play_num_trains(route_dist)
        self.add_points(game_manager.get_route_value(route_dist))
        self.player_board.add_edge(city1, city2, route_dist, cards_color)
        self.check_completed_tickets()
        return True

    def check_completed_tickets(self):
        for ticket, completed in self.tickets.items():
            if not completed and self.player_board.is_ticket_completed(ticket):
                logger.info('TICKET_COMPLETED!')
                self.complete_ticket(ticket)

    def choose_tickets(self, min_keep: int, tickets_num: int) -> Tuple[List[int], List[int]]:
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
