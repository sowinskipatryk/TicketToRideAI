from game_logic.enums.player_types import PlayerType

from game_logic.players.ai_player import AIPlayer
from game_logic.players.human_player import HumanPlayer
from game_logic.players.random_player import RandomPlayer


class PlayerFactory:
    def create_players(self, player_types, game_manager):
        return [self.create_player(index, player_type, game_manager) for index, player_type in enumerate(player_types)]

    @staticmethod
    def create_player(index, type_, game_manager):
        player_type = PlayerType(type_)
        if player_type == PlayerType.HUMAN:
            return HumanPlayer(index, game_manager)
        elif player_type == PlayerType.AI:
            return AIPlayer(index, game_manager)
        elif player_type == PlayerType.RANDOM:
            return RandomPlayer(index, game_manager)
        else:
            raise Exception("Invalid player type. Choose from: Human, AI, Random")
