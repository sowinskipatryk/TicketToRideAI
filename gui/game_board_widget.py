"""Game board visualization widget."""
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt, pyqtSignal, QPointF, QRectF
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor, QFont, QPainterPath
import networkx as nx
import math
from collections import defaultdict


class GameBoardWidget(QWidget):
    """Widget for displaying and interacting with the game board."""
    
    route_clicked = pyqtSignal(int)  # Emits link_id when route is clicked
    
    def __init__(self):
        super().__init__()
        self.graph = None
        self.claimed_routes = {}  # link_id -> player_color
        self.available_routes = set()  # link_ids that can be claimed
        self.node_positions = {}
        self.route_rects = {}  # link_id -> QRect for click detection
        self.selected_route = None
        
        self.setMinimumSize(1000, 700)
        self.setStyleSheet("background-color: #f0f0f0;")
    
    def set_graph(self, graph: nx.MultiGraph, cities: list, city_coordinates: dict = None):
        """Set the game graph and calculate positions.

        Args:
            graph: NetworkX graph representing the board
            cities: List of city names
            city_coordinates: Optional dict mapping city names to (lat, lon) tuples
        """
        self.graph = graph
        self.city_coordinates = city_coordinates  # Store for resize events

        # Use geographic coordinates if available
        if city_coordinates:
            # Convert lat/lon to screen coordinates
            # Note: longitude is x-axis, latitude is y-axis
            # Negate latitude since screen Y increases downward
            pos = {city: (lon, -lat) for city, (lat, lon) in city_coordinates.items() if city in graph.nodes()}
        else:
            # Fallback to spring layout
            # Calculate node positions using spring layout with much better spacing
            # Increase k parameter significantly for larger node separation
            # Use more iterations for better convergence
            num_nodes = len(graph.nodes())
            # Scale k based on number of nodes for optimal spacing
            k_value = max(5, math.sqrt(num_nodes) * 2)
            pos = nx.spring_layout(graph, k=k_value, iterations=200, seed=42)

        # Scale positions to widget size - use more of the available space
        if pos:
            min_x = min(p[0] for p in pos.values())
            max_x = max(p[0] for p in pos.values())
            min_y = min(p[1] for p in pos.values())
            max_y = max(p[1] for p in pos.values())
            
            # Use more of the widget space (reduce padding)
            padding = 50  # Reduced from 150
            scale_x = (self.width() - padding) / (max_x - min_x) if max_x != min_x else 1
            scale_y = (self.height() - padding) / (max_y - min_y) if max_y != min_y else 1
            # Use 95% of available space instead of 85%
            scale = min(scale_x, scale_y) * 0.95
            
            offset_x = (self.width() - (max_x - min_x) * scale) / 2
            offset_y = (self.height() - (max_y - min_y) * scale) / 2
            
            self.node_positions = {
                node: (
                    (pos[node][0] - min_x) * scale + offset_x,
                    (pos[node][1] - min_y) * scale + offset_y
                )
                for node in pos
            }
        else:
            self.node_positions = {}
        
        # Pre-calculate route groups for parallel routes (same cities, different colors)
        self._calculate_route_groups()
        
        self.update()
    
    def resizeEvent(self, event):
        """Handle widget resize - recalculate positions if graph is set."""
        super().resizeEvent(event)
        if self.graph:
            # Recalculate positions with new widget size
            # Preserve city_coordinates if they were set
            city_coords = getattr(self, 'city_coordinates', None)
            self.set_graph(self.graph, list(self.graph.nodes()), city_coords)
    
    def _calculate_route_groups(self):
        """Group routes by their endpoints for parallel route handling."""
        if not self.graph:
            self.route_groups = {}
            return
        
        # Group routes by (city1, city2) tuple (normalized order)
        groups = defaultdict(list)
        for u, v, data in self.graph.edges(data=True):
            link_id = data.get('link_id')
            if link_id is not None:
                # Normalize order (smaller city name first)
                key = tuple(sorted([u, v]))
                groups[key].append((link_id, data))
        
        # Store groups with multiple routes (parallel routes)
        self.route_groups = {k: v for k, v in groups.items() if len(v) > 1}
    
    def set_claimed_routes(self, claimed_routes: dict):
        """Update claimed routes.
        
        Args:
            claimed_routes: Dict mapping link_id to player_color
        """
        self.claimed_routes = claimed_routes
        self.update()
    
    def set_available_routes(self, route_ids: set):
        """Set which routes are available for claiming.
        
        Args:
            route_ids: Set of link_ids that can be claimed
        """
        self.available_routes = route_ids
        self.update()
    
    def paintEvent(self, event):
        """Paint the game board."""
        if not self.graph:
            painter = QPainter(self)
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "No game board loaded"
            )
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        self.route_rects.clear()
        
        # First pass: Draw unclaimed routes (background)
        # Group by route_id to handle parallel routes
        routes_by_endpoints = defaultdict(list)
        for u, v, data in self.graph.edges(data=True):
            link_id = data.get('link_id')
            if link_id is None:
                continue
            if u not in self.node_positions or v not in self.node_positions:
                continue
            key = tuple(sorted([u, v]))
            routes_by_endpoints[key].append((link_id, u, v, data))
        
        # Draw routes
        for (city1, city2), routes in routes_by_endpoints.items():
            num_parallel = len(routes)
            
            for idx, (link_id, u, v, data) in enumerate(routes):
                x1, y1 = self.node_positions[u]
                x2, y2 = self.node_positions[v]
                
                # Determine color and style
                claimed_by = self.claimed_routes.get(link_id)
                route_color = data.get('edge_color', 'gray')
                route_length = data.get('weight', 1)
                
                if claimed_by:
                    # Route is claimed - use player color (fully opaque and thicker)
                    color = self._get_player_color(claimed_by)
                    pen_width = 6  # Thicker for claimed routes
                    is_curved = False
                elif link_id in self.available_routes:
                    # Available route - use route color with highlight
                    base_color = self._get_route_color(route_color)
                    # Make it brighter/more visible
                    color = QColor(
                        min(255, base_color.red() + 50),
                        min(255, base_color.green() + 50),
                        min(255, base_color.blue() + 50),
                        220  # Semi-transparent
                    )
                    pen_width = 4
                    is_curved = num_parallel > 1
                else:
                    # Unavailable route - use route color with transparency
                    base_color = self._get_route_color(route_color)
                    color = QColor(base_color.red(), base_color.green(), base_color.blue(), 120)  # More transparent
                    pen_width = 3
                    is_curved = num_parallel > 1
                
                # Calculate offset for parallel routes (curve them)
                offset = 0
                if is_curved and num_parallel > 1:
                    # Offset parallel routes in a curve
                    offset_idx = idx - (num_parallel - 1) / 2
                    offset = offset_idx * 18  # 18 pixels per parallel route for better separation
                
                # Draw route line
                pen = QPen(color, pen_width)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                painter.setPen(pen)
                
                # Calculate midpoint and offset for parallel routes
                dx = x2 - x1
                dy = y2 - y1
                length = math.sqrt(dx*dx + dy*dy) if (dx != 0 or dy != 0) else 1
                perp_x = 0
                perp_y = 0
                
                if is_curved and offset != 0 and length > 0:
                    # Perpendicular offset for parallel routes
                    perp_x = -dy / length * offset
                    perp_y = dx / length * offset
                
                mid_x = (x1 + x2) / 2 + perp_x
                mid_y = (y1 + y2) / 2 + perp_y
                
                if is_curved and offset != 0:
                    # Draw curved line for parallel routes
                    path = QPainterPath()
                    path.moveTo(x1, y1)
                    path.quadTo(mid_x, mid_y, x2, y2)
                    painter.drawPath(path)
                else:
                    # Draw straight line
                    painter.drawLine(int(x1), int(y1), int(x2), int(y2))
                
                # Store rect for click detection (larger area, along the route)
                click_radius = 15
                # Store the route endpoints for better click detection
                self.route_rects[link_id] = {
                    'rect': (mid_x - click_radius, mid_y - click_radius, click_radius * 2, click_radius * 2),
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'mid_x': mid_x, 'mid_y': mid_y
                }
                
                # Draw route length label
                if not claimed_by:  # Only show length for unclaimed routes
                    painter.setFont(QFont("Arial", 8, QFont.Weight.Bold))  # Slightly smaller

                    # Background for text
                    text = str(route_length)
                    text_rect = painter.fontMetrics().boundingRect(text)
                    text_x = mid_x - text_rect.width() / 2
                    text_y = mid_y - text_rect.height() / 2

                    # Draw white background circle (smaller and more transparent)
                    bg_radius = max(text_rect.width(), text_rect.height()) / 2 + 2.5
                    painter.setBrush(QBrush(QColor(255, 255, 255, 200)))  # More transparent
                    painter.setPen(QPen(QColor(100, 100, 100), 1))  # Lighter border
                    painter.drawEllipse(
                        int(text_x - bg_radius + text_rect.width() / 2),
                        int(text_y - bg_radius + text_rect.height() / 2),
                        int(bg_radius * 2),
                        int(bg_radius * 2)
                    )

                    # Draw text
                    painter.setPen(QPen(QColor(60, 60, 60)))  # Slightly lighter black
                    painter.drawText(
                        int(text_x),
                        int(text_y + text_rect.height()),
                        text
                    )
        
        # Draw nodes (cities) on top
        painter.setFont(QFont("Arial", 9, QFont.Weight.Bold))
        for node, (x, y) in self.node_positions.items():
            # Draw city circle with white background
            painter.setBrush(QBrush(QColor(255, 255, 255)))
            painter.setPen(QPen(QColor(0, 0, 0), 2))
            radius = 10
            painter.drawEllipse(int(x - radius), int(y - radius), radius * 2, radius * 2)

            # Draw city name with background and outline for better contrast
            text_rect = painter.fontMetrics().boundingRect(node)
            text_x = int(x - text_rect.width() / 2)
            text_y = int(y - radius - 8)

            # White background for text (semi-transparent)
            painter.setBrush(QBrush(QColor(255, 255, 255, 220)))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRect(
                text_x - 3, text_y - text_rect.height() - 1,
                text_rect.width() + 6, text_rect.height() + 3
            )

            # Draw text with subtle outline for better readability
            # Draw outline
            painter.setPen(QPen(QColor(255, 255, 255), 3))
            painter.drawText(text_x, text_y, node)
            # Draw text
            painter.setPen(QPen(QColor(0, 0, 0)))
            painter.drawText(text_x, text_y, node)
    
    def mousePressEvent(self, event):
        """Handle mouse clicks on routes."""
        if not self.graph:
            return
        
        x, y = event.position().x(), event.position().y()
        
        # Check if clicked on a route (improved detection)
        # Check distance to route line, not just center point
        clicked_route = None
        min_distance = float('inf')
        
        for link_id, route_data in self.route_rects.items():
            if isinstance(route_data, dict):
                # New format with route endpoints
                rx, ry, rw, rh = route_data['rect']
                x1, y1 = route_data['x1'], route_data['y1']
                x2, y2 = route_data['x2'], route_data['y2']
            else:
                # Old format (tuple) - convert for compatibility
                rx, ry, rw, rh = route_data
                # Find the actual route to get endpoints
                for u, v, data in self.graph.edges(data=True):
                    if data.get('link_id') == link_id:
                        if u in self.node_positions and v in self.node_positions:
                            x1, y1 = self.node_positions[u]
                            x2, y2 = self.node_positions[v]
                        break
                else:
                    continue
            
            # First check if in bounding box
            if rx <= x <= rx + rw and ry <= y <= ry + rh:
                # Calculate distance from point to line segment
                dist = self._point_to_line_distance(x, y, x1, y1, x2, y2)
                if dist < min_distance and dist < 20:  # 20 pixel threshold
                    min_distance = dist
                    clicked_route = link_id
        
        if clicked_route and clicked_route in self.available_routes:
            self.selected_route = clicked_route
            self.route_clicked.emit(clicked_route)
            self.update()
    
    def _point_to_line_distance(self, px, py, x1, y1, x2, y2):
        """Calculate distance from point to line segment."""
        # Vector from line start to end
        dx = x2 - x1
        dy = y2 - y1
        
        if dx == 0 and dy == 0:
            # Line is a point
            return math.sqrt((px - x1)**2 + (py - y1)**2)
        
        # Parameter t for closest point on line
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
        
        # Closest point on line segment
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        
        # Distance from point to closest point
        return math.sqrt((px - closest_x)**2 + (py - closest_y)**2)
    
    def _get_player_color(self, player_color) -> QColor:
        """Get QColor for player color."""
        color_map = {
            'blue': QColor(0, 0, 255),
            'red': QColor(255, 0, 0),
            'green': QColor(0, 255, 0),
            'yellow': QColor(255, 255, 0),
            'black': QColor(0, 0, 0),
        }
        return color_map.get(str(player_color).lower(), QColor(128, 128, 128))
    
    def _get_route_color(self, route_color: str) -> QColor:
        """Get QColor for route color."""
        color_map = {
            'red': QColor(255, 0, 0),
            'blue': QColor(0, 0, 255),
            'green': QColor(0, 255, 0),
            'yellow': QColor(255, 255, 0),
            'black': QColor(0, 0, 0),
            'white': QColor(255, 255, 255),
            'pink': QColor(255, 192, 203),
            'orange': QColor(255, 165, 0),
            'grey': QColor(128, 128, 128),
            'gray': QColor(128, 128, 128),
        }
        return color_map.get(route_color.lower(), QColor(128, 128, 128))

