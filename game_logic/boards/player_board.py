import matplotlib.pyplot as plt
import networkx as nx

from typing import Tuple


class PlayerBoard:
    def __init__(self, player):
        self.player = player
        self.G = nx.Graph()

    def add_edge(self, city1: str, city2: str, route_dist: int, color: str) -> None:
        if city1 not in self.G:
            self.G.add_node(city1)
        if city2 not in self.G:
            self.G.add_node(city2)
        self.G.add_edge(city1, city2, weight=route_dist, edge_color=color)

    def is_ticket_completed(self, ticket: Tuple[str, str, int]) -> bool:
        start, end, _ = ticket

        if start not in self.G or end not in self.G:
            return False

        if not list(self.G.neighbors(start)):
            return False

        def dfs(node, visited):
            if node == end:
                return True
            visited.add(node)
            for neighbor in self.G.neighbors(node):
                if neighbor not in visited:
                    if dfs(neighbor, visited.copy()):
                        return True
            return False

        return dfs(start, set())

    def calculate_longest_path(self) -> int:
        def dfs(node_, visited):
            visited.add(node_)
            max_length = 0
            for neighbor in self.G.neighbors(node_):
                if neighbor not in visited:
                    length_ = 1 + dfs(neighbor, visited.copy())
                    if length_ > max_length:
                        max_length = length_
            return max_length

        longest = 0
        for node in self.G.nodes:
            if not list(self.G.neighbors(node)):
                continue
            length = dfs(node, set())
            if length > longest:
                longest = length

        return longest

    def draw_graph(self, pause_time: int = 30) -> None:
        plt.figure(figsize=(12, 6))

        pos = nx.spring_layout(self.G)
        nx.draw_networkx_nodes(self.G, pos, node_size=200)
        nx.draw_networkx_labels(self.G, pos, font_size=8)

        edge_labels = {(u, v): data["weight"] for u, v, data in self.G.edges(data=True)}
        for (u, v, data) in self.G.edges(data=True):
            nx.draw_networkx_edges(self.G, pos, edge_color=self.player.color.value, width=data['weight'])
            nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels, font_size=10)

        plt.axis("off")
        plt.ion()
        plt.show()
        plt.pause(pause_time)
        plt.close()
