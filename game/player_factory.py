from typing import List, Tuple, TYPE_CHECKING

from neat.nn.feed_forward import FeedForwardNetwork

from game.enums import PlayerType
from game.players.base_player import BasePlayer
from game.players.neat_player import NEATPlayer
from game.players.human_player import HumanPlayer
from game.players.random_player import RandomPlayer
from game.players.greedy_route_agent import GreedyRouteAgent
from game.players.ticket_focused_agent import TicketFocusedAgent
from game.players.card_hoarder_agent import CardHoarderAgent
from game.players.blocker_agent import BlockerAgent
from game.players.mcts_player import MCTSPlayer
from game.players.alphazero_player import AlphaZeroPlayer

from network.manager import load_network
from network.adapters.blank_adapter import BlankAdapter
from network.adapters.network_adapter import NetworkAdapter

if TYPE_CHECKING:
    from game.core import Game


class PlayerFactory:
    def create_players(self, player_types: List[str], game: 'Game', networks: List[FeedForwardNetwork] = None, ui_interface=None) -> Tuple[List[BasePlayer], BlankAdapter]:
        adapter = self.determine_adapter(player_types, game)
        players = []
        for index, player_type in enumerate(player_types):
            network = networks[index] if networks else None
            player = self.create_player(index, player_type, game, adapter, network, ui_interface=ui_interface)
            players.append(player)
        return players, adapter

    @staticmethod
    def determine_adapter(player_types: List[str], game: 'Game') -> BlankAdapter:
        if any(player_type == PlayerType.NEAT.value for player_type in player_types):
            return NetworkAdapter(game)
        else:
            return BlankAdapter()

    @staticmethod
    def create_player(index: int, type_: str, game, adapter: BlankAdapter, network: FeedForwardNetwork = None, ui_interface=None) -> BasePlayer:
        try:
            player_type = PlayerType(type_)
        except ValueError:
            raise ValueError(f"Invalid player type {type_}. Choose from: {[e.value for e in PlayerType]}")

        player_types = {
            PlayerType.NEAT: NEATPlayer,
            PlayerType.HUMAN: HumanPlayer,
            PlayerType.RANDOM: RandomPlayer,
            PlayerType.GREEDY: GreedyRouteAgent,
            PlayerType.TICKET_FOCUSED: TicketFocusedAgent,
            PlayerType.CARD_HOARDER: CardHoarderAgent,
            PlayerType.BLOCKER: BlockerAgent,
            PlayerType.MCTS: MCTSPlayer,
            PlayerType.ALPHAZERO: AlphaZeroPlayer,
            }

        player_class = player_types.get(player_type)

        if player_type == PlayerType.NEAT:
            if network is None:
                network = load_network()
            return player_class(index, game, adapter, network)
        elif player_type == PlayerType.HUMAN:
            return player_class(index, game, adapter, ui_interface=ui_interface)

        return player_class(index, game, adapter)
