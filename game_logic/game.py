import random
from typing import List
from neat.nn.feed_forward import FeedForwardNetwork

from game_logic.game_logger import logger
from game_logic.enums.game_states import GameState
from game_logic.player_factory import PlayerFactory
from game_logic.players.base_player import BasePlayer
from game_logic.boards.game_board import GameBoard
from game_logic.ticket_deck import TicketDeck
from game_logic.train_card_manager import TrainCardManager
from game_logic.config_factory import ConfigFactory


class Game:
    MIN_PLAYERS = 2
    MAX_PLAYERS = 5

    def __init__(self, player_types: List[str], version: str, networks: List[FeedForwardNetwork] = None) -> None:
        self.players_num = len(player_types)

        self.config_factory = ConfigFactory()
        self.config = self.config_factory.get_config(version)
        self.version = version

        if not self.MIN_PLAYERS <= self.players_num <= self.MAX_PLAYERS:
            raise ValueError(f'This game is designed for {self.MIN_PLAYERS}-{self.MAX_PLAYERS} players.')

        self.game_state = GameState.INIT
        logger.info(self.game_state)

        self.board = GameBoard(self)
        self.ticket_deck = TicketDeck(self)

        self.player_factory = PlayerFactory()
        self.players, self.adapter = self.player_factory.create_players(player_types, self, networks)

        self.train_card_manager = TrainCardManager(self)

        self.current_player_id = random.randrange(0, self.players_num)
        self.total_moves = 0
        self.completed_moves = 0

        self.last_player = None
        self.winner = None

    def move_to_next_player(self) -> None:
        self.current_player_id = (self.current_player_id + 1) % self.players_num

    def get_current_player(self) -> BasePlayer:
        return self.players[self.current_player_id]

    def last_round_condition(self, player: BasePlayer) -> bool:
        return player.get_trains_num() <= self.config.MIN_TRAIN_FIGURES_NUM

    def play(self, max_moves: int = 0) -> List[int]:
        self.ticket_deck.set_ticket_pile_num_adapter()
        self.game_state = GameState.RUNNING
        logger.info(self.game_state)

        for player in self.players:
            player.draw_initial_train_cards(self.config.STARTING_HAND_SIZE)
            player.draw_tickets(num_tickets=self.config.INITIAL_TICKETS_DEALT_NUM,
                                min_keep=self.config.INITIAL_TICKETS_TO_KEEP_NUM)

        while self.game_state != GameState.LAST_ROUND:
            if max_moves and self.total_moves > max_moves:
                break
            self.total_moves += 1
            current_player = self.players[self.current_player_id]
            move_completed = current_player.play_turn()
            if move_completed:
                self.completed_moves += 1
            else:
                self.player_stats[current_player.player_id]['invalid_actions'] += 1
            self.log_game_state()
            if self.last_round_condition(current_player):
                self.last_player = current_player
                self.game_state = GameState.LAST_ROUND
            self.move_to_next_player()

        while self.game_state != GameState.FINISHED:
            if max_moves and self.total_moves > max_moves:
                break
            self.total_moves += 1
            current_player = self.players[self.current_player_id]
            move_completed = current_player.play_turn()
            if move_completed:
                self.completed_moves += 1
            else:
                self.player_stats[current_player.player_id]['invalid_actions'] += 1
            self.log_game_state()
            if current_player is self.last_player:
                self.game_state = GameState.FINISHED
            else:
                self.move_to_next_player()

        self.score_player_tickets()
        self.score_longest_path()

        self.determine_winner()
        logger.info(f'{self.winner} {self.winner.tickets} won!')
        logger.info(f'completed moves: {self.completed_moves}')
        logger.info(f'total moves: {self.total_moves}')

        self.game_summary()

        return [player.score for player in self.players]

    def score_player_tickets(self) -> None:
        for player in self.players:
            player.score_tickets()

    def score_longest_path(self) -> None:
        if not hasattr(self.config, 'LONGEST_ROUTE_BONUS'):
            return

        max_value = float('-inf')
        best_players = []

        for player in self.players:
            value = player.player_board.calculate_longest_path()

            if value > max_value:
                max_value = value
                best_players = [player]
            elif value == max_value:
                best_players.append(player)

        for player in best_players:
            player.set_longest_path()
            player.add_points(self.config.LONGEST_ROUTE_BONUS)

    def determine_winner(self) -> None:
        sorted_players = sorted(self.players,
                                key=lambda player: (player.get_score(), sum(player.get_ticket_values()),
                                                    player.longest_path),
                                reverse=True)
        self.winner = sorted_players[0]
        logger.info(f'{self.winner} wins!')
        # self.winner.player_board.draw_graph()
        logger.info('winner tickets:', self.winner.tickets)
        # self.game_manager.board.draw_possession_graph()

    def log_game_state(self):
        # logger.debug(f'face up pile: {self.game_manager.train_card_manager.get_face_up_cards()}')
        # logger.debug(f'draw pile: {len(self.game_manager.train_card_manager.get_draw_pile())}')
        # logger.debug(f'discard pile: {len(self.game_manager.train_card_manager.get_discard_pile())}')
        logger.info(f'ticket deck: {self.ticket_deck.tickets_left}')
        for player in self.players:
            logger.info(f'{player} trains: {player.trains_remaining}')

    def game_summary(self):
        for player in self.players:
            logger.info(player)
            logger.info(f'score: {player.score}')
            logger.info(f'tickets: {player.tickets}')
            logger.info(f'trains remaining: {player.trains_remaining}')
            logger.info(f'longest path: {player.longest_path}')

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

    def get_route_value(self, route_length: int) -> int:
        return self.config.ROUTE_VALUES[route_length]
