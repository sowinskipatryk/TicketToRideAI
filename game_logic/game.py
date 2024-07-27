import random
from typing import List

from game_logic.game_manager import GameManager
from game_logic.game_config import GameConfigAmerica
from game_logic.game_logger import logger
from game_logic.enums.game_states import GameState
from game_logic.player_factory import PlayerFactory
from game_logic.players.base_player import BasePlayer


class Game:
    def __init__(self, player_types: List[str]) -> None:

        self.players_num = len(player_types)

        if not 1 < self.players_num < 6:
            raise ValueError('This game is designed for 2-5 players.')

        self.game_state = GameState.INIT
        logger.info(self.game_state)

        self.player_factory = PlayerFactory()
        self.players = self.player_factory.create_players(player_types)
        self.game_manager = GameManager(self.players_num)
        self.current_player_id = random.randrange(0, self.players_num)
        self.total_moves = 0
        self.completed_moves = 0

        self.last_player = None
        self.winner = None
        self.route_values = GameConfigAmerica.ROUTE_VALUES

    def move_to_next_player(self) -> None:
        self.current_player_id = (self.current_player_id + 1) % self.players_num

    def get_current_player(self) -> BasePlayer:
        return self.players[self.current_player_id]

    @staticmethod
    def last_round_condition(player: BasePlayer) -> bool:
        return player.get_trains_num() <= GameConfigAmerica.MIN_TRAIN_FIGURES_NUM

    def play(self) -> None:
        self.game_state = GameState.RUNNING
        logger.info(self.game_state)

        for player in self.players:
            player.draw_initial_train_cards(self.game_manager, GameConfigAmerica.STARTING_HAND_SIZE)
            player.draw_tickets(self.game_manager,
                                num_tickets=GameConfigAmerica.INITIAL_TICKETS_DEALT_NUM,
                                min_keep=GameConfigAmerica.INITIAL_TICKETS_TO_KEEP_NUM)

        while self.game_state != GameState.LAST_ROUND:
            self.total_moves += 1
            current_player = self.players[self.current_player_id]
            move_completed = current_player.play_turn(self.game_manager)
            if move_completed:
                self.completed_moves += 1
            self.log_game_state()
            if self.last_round_condition(current_player):
                self.last_player = current_player
                self.game_state = GameState.LAST_ROUND
            self.move_to_next_player()

        while self.game_state != GameState.FINISHED:
            self.total_moves += 1
            current_player = self.players[self.current_player_id]
            move_completed = current_player.play_turn(self.game_manager)
            if move_completed:
                self.completed_moves += 1
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

    def score_player_tickets(self) -> None:
        for player in self.players:
            player.score_tickets()

    def score_longest_path(self) -> None:
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
            player.add_points(GameConfigAmerica.LONGEST_ROUTE_BONUS)

    def determine_winner(self) -> None:
        sorted_players = sorted(self.players,
                                key=lambda player: (player.get_score(),
                                                    sum(player.get_ticket_values()),
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
        logger.info(f'ticket deck: {self.game_manager.ticket_deck.tickets_left()}')
        for player in self.players:
            logger.info(f'{player} trains: {player.trains_remaining}')

    def game_summary(self):
        for player in self.players:
            logger.info(player)
            logger.info(f'score: {player.score}')
            logger.info(f'tickets: {player.tickets}')
            logger.info(f'trains remaining: {player.trains_remaining}')
            logger.info(f'longest path: {player.longest_path}')
