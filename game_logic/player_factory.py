from enums.player_types import PlayerType

from players.ai_player import AIPlayer
from players.human_player import HumanPlayer
from players.random_player import RandomPlayer


class PlayerFactory:
    @staticmethod
    def create_player(index, player_type):
        if player_type == PlayerType.HUMAN:
            return HumanPlayer(index)
        elif player_type == PlayerType.AI:
            return AIPlayer(index)
        elif player_type == PlayerType.RANDOM:
            return RandomPlayer(index)
        else:
            raise Exception("Invalid player type. Choose from: Human, AI, Random")
