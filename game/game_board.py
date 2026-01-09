import matplotlib.pyplot as plt
import networkx as nx

from collections import defaultdict
from typing import Dict, Union, Tuple, TYPE_CHECKING

from game.data_loader import load_routes, load_cities
from game.game_logger import logger

if TYPE_CHECKING:
    from game.core import Game


class GameBoard:
    """Manages the game board, routes, and path calculations.
    
    Uses NetworkX to represent the board as a graph and handles route
    claiming, validation, and path finding operations.
    """
    def __init__(self, game: 'Game'):
        """Initialize the game board.
        
        Args:
            game: Reference to the Game instance
        """
        self.game = game
        self.routes = load_routes(self.game.version)
        self.cities = load_cities(self.game.version)
        self.G = nx.MultiGraph()
        self._fill_graph()

    def get_route_links_num(self) -> int:
        return self.G.number_of_edges()

    def _fill_graph(self) -> None:
        self.G.add_nodes_from(self.cities)

        link_id = 0
        for route_id, route in enumerate(self.routes):
            for color in route['colors']:
                self.G.add_edge(route['from'], route['to'],
                                route_id=route_id,
                                link_id=link_id,
                                weight=route['length'],
                                edge_color=color,
                                claimed_by=None)
                link_id += 1

    def claim_route(self, link_id: int, player_color: str) -> bool:
        """Claim a route for a player.
        
        Args:
            link_id: The link ID of the route to claim
            player_color: The color of the player claiming the route
            
        Returns:
            True if route was successfully claimed, False otherwise
        """
        for u, v, data in self.G.edges(data=True):
            if data['link_id'] == link_id:
                data['claimed_by'] = player_color
                # Invalidate cache when route is claimed
                self.invalidate_path_cache()
                return True
        return False

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

    def validate_route(self, player, data) -> bool:
        """Validate if a player can claim a route.
        
        Args:
            player: Player color attempting to claim the route
            data: Route data dictionary containing route_id
            
        Returns:
            True if route can be claimed, False otherwise
        """
        owners = self.get_route_owners(data['route_id'])

        if not any(owners):
            return True

        if player in owners:
            logger.info('You already claimed one link for this route!')
            return False

        if all(owners):
            logger.info('All links are claimed for this route!')
            return False

        if not self.game.config.WILD_CARD_RESTRICTION or self.game.players_num > 3:
            return True

        logger.info('You can only claim one link for each route in this game configuration!')
        return False

    def get_route_data(self, link_id: int) -> Tuple[str, str, Dict[str, Union[bool, str, int]]]:
        for u, v, data in self.G.edges(data=True):
            if data['link_id'] == link_id:
                return u, v, data

    def player_subgraph(self, player_color):
        G = nx.Graph()
        for u, v, data in self.G.edges(data=True):
            if data["claimed_by"] == player_color:
                G.add_edge(u, v, weight=data["weight"], edge_color=data["edge_color"])
        return G

    def is_ticket_completed(self, player_color, ticket) -> bool:
        """Check if a player has completed a destination ticket.
        
        Args:
            player_color: Color of the player to check
            ticket: Ticket object with city_from and city_to
            
        Returns:
            True if there's a path between the ticket cities, False otherwise
        """
        G = self.player_subgraph(player_color)
        if ticket.city_from not in G or ticket.city_to not in G:
            return False
        return nx.has_path(G, ticket.city_from, ticket.city_to)

    def calculate_longest_path(self, player_color) -> int:
        """Calculate the longest continuous path for a player.
        
        Uses depth-first search to find the longest path. Results are cached
        per player to avoid redundant calculations.
        
        Args:
            player_color: The color of the player to calculate path for
            
        Returns:
            The length of the longest path
        """
        # Cache key for this player's subgraph
        # Use a hash of claimed routes to invalidate cache when board changes
        cache_key = (player_color, self._get_board_state_hash())
        
        # Check cache (simple in-memory cache per game instance)
        if not hasattr(self, '_longest_path_cache'):
            self._longest_path_cache = {}
        
        if cache_key in self._longest_path_cache:
            return self._longest_path_cache[cache_key]
        
        G = self.player_subgraph(player_color)

        def dfs(node, visited_edges):
            max_length = 0
            for neighbor in G.neighbors(node):
                edge = tuple(sorted((node, neighbor)))
                if edge in visited_edges:
                    continue
                visited_edges.add(edge)
                edge_length = G[node][neighbor]["weight"]
                max_length = max(max_length, edge_length + dfs(neighbor, visited_edges))
                visited_edges.remove(edge)
            return max_length

        longest = 0
        for node in G.nodes:
            longest = max(longest, dfs(node, set()))
        
        # Cache the result
        self._longest_path_cache[cache_key] = longest
        return longest
    
    def _get_board_state_hash(self) -> int:
        """Get a hash of the current board state for cache invalidation.
        
        Returns:
            Hash value representing current board state
        """
        # Create a simple hash based on claimed routes
        claimed_routes = tuple(
            sorted((u, v, data.get('claimed_by'))
                  for u, v, data in self.G.edges(data=True)
                  if data.get('claimed_by') is not None)
        )
        return hash(claimed_routes)
    
    def invalidate_path_cache(self):
        """Invalidate the longest path cache (call when routes are claimed)."""
        if hasattr(self, '_longest_path_cache'):
            self._longest_path_cache.clear()

    def count_claimed_routes(self, player_color) -> int:
        G = self.player_subgraph(player_color)
        return G.number_of_edges()

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
                                           or not self.game.config.WILD_CARD_RESTRICTION
                                           or self.game.players_num > 3):
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
