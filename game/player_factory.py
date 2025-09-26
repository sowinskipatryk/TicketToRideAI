from typing import List, Tuple, TYPE_CHECKING

from neat.nn.feed_forward import FeedForwardNetwork

from game.enums import PlayerType
from game.players.base_player import BasePlayer
from game.players.neat_player import NEATPlayer
from game.players.human_player import HumanPlayer
from game.players.random_player import RandomPlayer

from network.manager import load_network
from network.adapters.base_adapter import BaseAdapter
from network.adapters.network_adapter import NetworkAdapter

if TYPE_CHECKING:
    from game.core import Game


class PlayerFactory:
    def create_players(self, player_types: List[str], game_instance: 'Game', networks: List[FeedForwardNetwork]) -> Tuple[List[BasePlayer], BaseAdapter]:
        adapter = self.determine_adapter(player_types, game_instance)
        players = []
        for index, player_type in enumerate(player_types):
            network = networks[index] if networks else None
            player = self.create_player(index, player_type, game_instance, adapter, network)
            players.append(player)
        return players, adapter

    @staticmethod
    def determine_adapter(player_types: List[str], game_instance: 'Game') -> BaseAdapter:
        if any(player_type == PlayerType.NEAT.value for player_type in player_types):
            return NetworkAdapter(game_instance)
        else:
            return BaseAdapter()

    @staticmethod
    def create_player(index: int, type_: str, game_instance, adapter: BaseAdapter, network: FeedForwardNetwork) -> BasePlayer:
        try:
            player_type = PlayerType(type_)
        except ValueError:
            raise ValueError(f"Invalid player type {type_}. Choose from: Human, AI, Random")

        player_types = {
            PlayerType.NEAT: NEATPlayer,
            PlayerType.HUMAN: HumanPlayer,
            PlayerType.RANDOM: RandomPlayer
            }

        player_class = player_types.get(player_type)

        if player_type == PlayerType.NEAT:
            if network is None:
                network = load_network()
            return player_class(index, game_instance, adapter, network)

        return player_class(index, game_instance, adapter)
