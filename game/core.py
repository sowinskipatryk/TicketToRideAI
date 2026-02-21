import random
from typing import List, Dict
from neat.nn.feed_forward import FeedForwardNetwork

from game.game_logger import logger
from game.enums import GameState
from game.player_factory import PlayerFactory
from game.players.base_player import BasePlayer
from game.game_board import GameBoard
from game.ticket_deck import TicketDeck, Ticket
from game.train_card_manager import TrainCardManager
from game.config import ConfigFactory


class Game:
    """Main game class that manages game state and turn flow.
    
    Handles player turns, route claiming, card drawing, ticket management,
    and game completion logic.
    """
    MIN_PLAYERS = 2
    MAX_PLAYERS = 5

    def __init__(self, player_types: List[str], version: str, networks: List[FeedForwardNetwork] = None, ui_interface=None) -> None:
        """Initialize a new game.
        
        Args:
            player_types: List of player type strings ('Human', 'NEAT', 'Random')
            version: Game version ('USA', 'Europe')
            networks: Optional list of NEAT networks for NEAT players
            ui_interface: Optional UI interface for human players (GUI or CLI)
            
        Raises:
            ValueError: If player count is invalid or version is unsupported
        """
        self.players_num = len(player_types)

        if not self.MIN_PLAYERS <= self.players_num <= self.MAX_PLAYERS:
            raise ValueError(f'This game is designed for {self.MIN_PLAYERS}-{self.MAX_PLAYERS} players.')

        self.config = ConfigFactory.create(version)
        self.version = version

        self.game_state = GameState.INIT
        logger.info(self.game_state)

        self.board = GameBoard(self)
        self.ticket_deck = TicketDeck(self)

        self.player_factory = PlayerFactory()
        self.players = self.player_factory.create_players(player_types, self, networks, ui_interface=ui_interface)

        self.train_card_manager = TrainCardManager(self)

        self.current_player_id = random.randrange(0, self.players_num)

        self.last_player = None
        self.winner = None

        # Game statistics dictionary - tracks move counts and game results
        # Note: GameStats class exists but is currently unused; stats are tracked directly here
        self.stats = {"invalid_moves": [0 for _ in range(self.players_num)],
                      "completed_moves": [0 for _ in range(self.players_num)],
                      "total_moves": [0 for _ in range(self.players_num)]}

        if hasattr(self.config, 'LONGEST_ROUTE_BONUS'):
            self.stats['longest_path_length'] = [0 for _ in range(self.players_num)]

    def _run_turn(self, max_moves: int) -> bool:
        """Execute the current player's turn and track stats.

        Returns:
            False if max_moves limit was reached before the turn, True otherwise.
        """
        if max_moves and sum(self.stats['total_moves']) >= max_moves:
            return False
        current_player = self.players[self.current_player_id]
        self.stats['total_moves'][self.current_player_id] += 1
        move_completed = current_player.play_turn()
        if move_completed:
            self.stats['completed_moves'][current_player.player_id] += 1
        else:
            self.stats['invalid_moves'][current_player.player_id] += 1
        self.log_game_state()
        return True

    def move_to_next_player(self) -> None:
        self.current_player_id = (self.current_player_id + 1) % self.players_num

    def get_current_player(self) -> BasePlayer:
        return self.players[self.current_player_id]

    def last_round_condition(self, player: BasePlayer) -> bool:
        return player.get_trains_num() <= self.config.MIN_TRAIN_FIGURES

    def _deal_initial(self) -> None:
        """Deal initial cards and tickets to all players, transition to RUNNING state."""
        self.game_state = GameState.RUNNING
        logger.info(self.game_state)
        for player in self.players:
            player.draw_initial_train_cards(self.config.NUM_TRAIN_CARDS_DEALT_INIT)
            player.draw_tickets(num_tickets=self.config.NUM_TICKETS_DEALT_INIT,
                                min_keep=self.config.MIN_TICKETS_KEPT_INIT)

    def play(self, max_moves: int = 0) -> Dict:
        """Run the game until completion.

        Args:
            max_moves: Maximum number of moves before stopping (0 = no limit)

        Returns:
            Dictionary containing game statistics:
            - invalid_moves: List of invalid move counts per player
            - completed_moves: List of completed move counts per player
            - total_moves: List of total move counts per player
            - completed_tickets: List of completed ticket counts per player
            - total_tickets: List of total ticket counts per player
            - trains_remaining: List of remaining train counts per player
            - longest_path_owner: List of longest path flags per player
            - score: List of final scores per player
            - claimed_routes: List of claimed route counts per player
            - longest_path_length: List of longest path lengths (if applicable)
        """
        self._deal_initial()

        while self.game_state != GameState.LAST_ROUND:
            if not self._run_turn(max_moves):
                break
            current_player = self.players[self.current_player_id]
            if self.last_round_condition(current_player):
                self.last_player = current_player
                self.game_state = GameState.LAST_ROUND
            self.move_to_next_player()

        while self.game_state != GameState.FINISHED:
            current_player = self.players[self.current_player_id]
            if not self._run_turn(max_moves):
                break
            if current_player.player_id == self.last_player.player_id:
                self.game_state = GameState.FINISHED
            else:
                self.move_to_next_player()

        self.score_player_tickets()
        self.score_longest_path()

        self.determine_winner()
        logger.info(f'{self.winner} {self.winner.tickets} won!')
        logger.info(f"completed moves: {self.stats['completed_moves']}")
        logger.info(f"total moves: {self.stats['total_moves']}")

        self.game_summary()

        self.stats['completed_tickets'] = [sum(player.tickets.values()) for player in self.players]
        self.stats['total_tickets'] = [len(player.tickets) for player in self.players]
        self.stats['trains_remaining'] = [player.trains_remaining for player in self.players]
        self.stats['longest_path_owner'] = [player.longest_path for player in self.players]
        self.stats['score'] = [player.score for player in self.players]
        self.stats['claimed_routes'] = [self.board.count_claimed_routes(player.color) for player in self.players]

        self.print_game_stats()
        return self.stats

    def score_player_tickets(self) -> None:
        for player in self.players:
            player.score_tickets()

    def score_longest_path(self) -> None:
        if not hasattr(self.config, 'LONGEST_ROUTE_BONUS'):
            return

        max_value = float('-inf')
        best_players = []

        for player_id, player in enumerate(self.players):
            value = self.board.calculate_longest_path(player.color)
            self.stats['longest_path_length'][player_id] = value

            if value > max_value:
                max_value = value
                best_players = [player]
            elif value == max_value:
                best_players.append(player)

        for player in best_players:
            player.set_longest_path()
            player.add_points(self.config.LONGEST_ROUTE_BONUS)

    def determine_winner(self) -> None:
        """Determine the game winner based on score, ticket values, and longest path.
        
        Sets self.winner to the player with the highest score. In case of ties,
        uses ticket values and longest path as tiebreakers.
        """
        sorted_players = sorted(self.players,
                                key=lambda player: (player.get_score(), sum(player.get_ticket_values()),
                                                    player.longest_path),
                                reverse=True)
        self.winner = sorted_players[0]
        logger.info(f'{self.winner} wins!')
        logger.info(f'winner tickets: {self.winner.tickets}')

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

    def deal_tickets(self, num_tickets: int) -> List[Ticket]:
        dealt_tickets: List[Ticket] = []
        while len(dealt_tickets) < num_tickets:
            ticket = self.ticket_deck.remove()
            if ticket is None:
                break
            dealt_tickets.append(ticket)
        return dealt_tickets

    def get_route_value(self, route_length: int) -> int:
        return self.config.ROUTE_VALUES[route_length]

    def print_game_stats(self):
        logger.info(f"completed_moves: {self.stats['completed_moves']}")
        logger.info(f"invalid_moves: {self.stats['invalid_moves']}")
        logger.info(f"total_moves: {self.stats['total_moves']}")
        logger.info(f"completed_tickets: {self.stats['completed_tickets']}")
        logger.info(f"total_tickets: {self.stats['total_tickets']}")
        logger.info(f"trains_remaining: {self.stats['trains_remaining']}")
        logger.info(f"claimed_routes: {self.stats['claimed_routes']}")
        if hasattr(self.config, 'LONGEST_ROUTE_BONUS'):
            logger.info(f"longest_path_length: {self.stats['longest_path_length']}")
            logger.info(f"longest_path_owner: {self.stats['longest_path_owner']}")
        logger.info(f"score: {self.stats['score']}")
