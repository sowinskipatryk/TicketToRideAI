import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, Union, Tuple

from game_logic.utils import load_routes, load_cities


class GameBoard:
    def __init__(self):
        super().__init__()
        self.routes = load_routes()
        self.cities = load_cities()
        self.G = nx.MultiGraph()
        self._fill_graph()

    def get_routes_num(self):
        return len(self.routes)
    def _fill_graph(self) -> None:
        self.G.add_nodes_from(self.cities)
        for route_id, route in enumerate(self.routes):
            self.G.add_edge(route[0], route[1],
                            route_id=route_id,
                            weight=int(route[2]),
                            edge_colors=route[3:],
                            claimed_by=[None for _ in range(len(route[3:]))])

    def claim_route(self, route_id, cards_color, player_color):
        for u, v, data in self.G.edges(data=True):
            if data['route_id'] == route_id:
                if cards_color not in data['edge_colors']:
                    route_pos = next(index for index, value in enumerate(data['claimed_by']) if value is None)
                else:
                    route_pos = data['edge_colors'].index(cards_color)
                data['claimed_by'][route_pos] = player_color
                return True

    def get_route_data(self, route_id: int) -> Tuple[str, str, Dict[str, Union[bool, str, int]]]:
        for u, v, data in self.G.edges(data=True):
            if data['route_id'] == route_id:
                return u, v, data

    def draw_possession_graph(self, pause_time=30):
        plt.figure(figsize=(18, 8))

        pos = nx.spring_layout(self.G)

        nx.draw_networkx_nodes(self.G, pos, node_size=700)

        nx.draw_networkx_labels(self.G, pos)

        ax = plt.gca()
        colors = []
        edge_labels = {}
        for (u, v, data) in self.G.edges(data=True):
            for i in range(len(data['edge_colors'])):
                if not data['claimed_by'][i]:
                    colors.append('grey')
                else:
                    colors.append(data['claimed_by'][i].value)

                rad = 0.1 * (i + 1)
                edge = nx.draw_networkx_edges(self.G, pos, edgelist=[(u, v)], edge_color=[colors[-1]], width=2, ax=ax,
                                              connectionstyle=f"arc3,rad={rad}")
                edge_labels[(u, v, i)] = data['weight']

        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels)

        plt.axis("off")
        plt.ion()
        plt.show()
        plt.pause(pause_time)
        plt.close()

    def draw_available_moves_graph(self, pause_time=30):
        plt.figure(figsize=(18, 8))

        plt.gcf().set_facecolor('#aaa')

        pos = nx.spring_layout(self.G)

        nx.draw_networkx_nodes(self.G, pos, node_size=700)

        nx.draw_networkx_labels(self.G, pos)

        ax = plt.gca()
        edge_labels = {}
        for (u, v, key, data) in self.G.edges(keys=True, data=True):
            if not data['claimed_by'][key]:
                rad = 0.1 * (key + 1)  # Adjust the curvature radius
                edge = nx.draw_networkx_edges(self.G, pos, edgelist=[(u, v)], edge_color=data["edge_colors"][key], width=2, ax=ax,
                                              connectionstyle=f"arc3,rad={rad}")
                edge_labels[(u, v, key)] = f"#{data['route_id']} L{data['weight']}"

        edge_label_objects = nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels)

        for _, text in edge_label_objects.items():
            text.set_bbox(dict(facecolor='none', edgecolor='none'))

        plt.axis("off")
        plt.ion()
        plt.show()
        plt.pause(pause_time)
        plt.close()
