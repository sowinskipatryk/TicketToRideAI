from game_logic.enums.player_types import PlayerType

from game_logic.players.ai_player import AIPlayer
from game_logic.players.human_player import HumanPlayer
from game_logic.players.random_player import RandomPlayer


class PlayerFactory:
    def create_players(self, player_types):
        return [self.create_player(index, player_type) for index, player_type in enumerate(player_types)]

    @staticmethod
    def create_player(index, type_):
        player_type = PlayerType(type_)
        if player_type == PlayerType.HUMAN:
            return HumanPlayer(index)
        elif player_type == PlayerType.AI:
            return AIPlayer(index)
        elif player_type == PlayerType.RANDOM:
            return RandomPlayer(index)
        else:
            raise Exception("Invalid player type. Choose from: Human, AI, Random")
