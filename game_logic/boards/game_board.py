import matplotlib.pyplot as plt
import networkx as nx

from collections import defaultdict
from typing import Dict, Union, Tuple, TYPE_CHECKING

from game_logic.game_logger import logger
from game_logic.utils import load_routes, load_cities

if TYPE_CHECKING:
    from game_logic.game import Game


class GameBoard:
    def __init__(self, game_instance: 'Game'):
        self.game_instance = game_instance
        self.routes = load_routes(self.game_instance.version)
        self.cities = load_cities(self.game_instance.version)
        self.G = nx.MultiGraph()
        self._fill_graph()

    def get_route_links_num(self) -> int:
        return self.G.number_of_edges()

    def _fill_graph(self) -> None:
        self.G.add_nodes_from(self.cities)

        link_id = 0
        for route_id, route in enumerate(self.routes):
            colors = route[3:]
            for color in colors:
                self.G.add_edge(route[0], route[1],
                                route_id=route_id,
                                link_id=link_id,
                                weight=int(route[2]),
                                edge_color=color,
                                claimed_by=None)
                link_id += 1

    def claim_route(self, link_id: int, player_color: str) -> bool:
        for u, v, data in self.G.edges(data=True):
            if data['link_id'] == link_id:
                data['claimed_by'] = player_color
                return True

    def get_route_owners(self, route_id):
        claimed_by = []
        for u, v, data in self.G.edges(data=True):
            if data['route_id'] == route_id:
                claimed_by.append(data['claimed_by'])
        return claimed_by

    def get_link_owner(self, link_id):
        for u, v, data in self.G.edges(data=True):
            if data['link_id'] == link_id:
                return data['claimed_by']

    def validate_route(self, player, data):
        owners = self.get_route_owners(data['route_id'])

        if not any(owners):
            return True

        if player in owners:
            logger.info('You already claimed one link for this route!')
            return

        if all(owners):
            logger.info('All links are claimed for this route!')
            return

        if not self.game_instance.config.WILD_CARD_RESTRICTION or self.game_instance.players_num > 3:
            return True

        logger.info('You can only claim one link for each route in this game configuration!')

    def get_route_data(self, link_id: int) -> Tuple[str, str, Dict[str, Union[bool, str, int]]]:
        for u, v, data in self.G.edges(data=True):
            if data['link_id'] == link_id:
                return u, v, data

    def draw_possession_graph(self, pause_time: int = 30) -> None:
        plt.figure(figsize=(18, 8))

        pos = nx.spring_layout(self.G)
        nx.draw_networkx_nodes(self.G, pos, node_size=700)
        nx.draw_networkx_labels(self.G, pos, font_size=11)

        ax = plt.gca()
        for (u, v, i, data) in self.G.edges(keys=True, data=True):
            if not data['claimed_by']:
                color = 'grey'
            else:
                color = data['claimed_by'].value

            rad = 0.1 * (i + 1)
            edge = nx.draw_networkx_edges(self.G, pos, edgelist=[(u, v)], edge_color=color, width=2, ax=ax,
                                          connectionstyle=f"arc3,rad={rad}")

        plt.axis("off")
        plt.ion()
        plt.show()
        plt.pause(pause_time)
        plt.close()

    def draw_available_moves_graph(self, pause_time: int = 30) -> None:
        plt.figure(figsize=(18, 8))
        plt.gcf().set_facecolor('#aaa')

        pos = nx.spring_layout(self.G)
        nx.draw_networkx_nodes(self.G, pos, node_size=700)
        nx.draw_networkx_labels(self.G, pos)

        ax = plt.gca()

        route_owners = defaultdict(list)
        for (u, v, i, data) in self.G.edges(keys=True, data=True):
            route_owners[(u, v)].append(data['claimed_by'])

            if data['claimed_by']:
                rad = 0.1 * (i + 1)
                edge = nx.draw_networkx_edges(self.G, pos, edgelist=[(u, v)], edge_color=data['claimed_by'].value,
                                              width=6, ax=ax, connectionstyle=f"arc3,rad={rad}")

        edge_labels = {}
        for (u, v, i, data) in self.G.edges(keys=True, data=True):
            rad = 0.1 * (i + 1)

            if not data['claimed_by'] and (not any(route_owners[(u, v)])
                                           or not self.game_instance.config.WILD_CARD_RESTRICTION
                                           or self.game_instance.players_num > 3):
                edge = nx.draw_networkx_edges(self.G, pos, edgelist=[(u, v)], edge_color=data["edge_color"],
                                              width=2, ax=ax, connectionstyle=f"arc3,rad={rad}")
                edge_labels[(u, v, i)] = f"#{data['link_id']} L{data['weight']}"

        edge_label_objects = nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels)

        for _, text in edge_label_objects.items():
            text.set_bbox(dict(facecolor='none', edgecolor='none'))

        plt.axis("off")
        plt.ion()
        plt.show()
        plt.pause(pause_time)
        plt.close()
