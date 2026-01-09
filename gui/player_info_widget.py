"""Player information display widget."""
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QGroupBox, QGridLayout
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor


class PlayerInfoWidget(QWidget):
    """Widget displaying player information."""
    
    def __init__(self):
        super().__init__()
        self.player_labels = {}
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI."""
        layout = QVBoxLayout(self)
        
        title = QLabel("Players")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        layout.addWidget(title)
        
        # Player info will be added dynamically
        self.players_group = QGroupBox()
        self.players_layout = QVBoxLayout()
        self.players_group.setLayout(self.players_layout)
        layout.addWidget(self.players_group)
        
        layout.addStretch()
    
    def update_players(self, players_data: list):
        """Update player information.
        
        Args:
            players_data: List of dicts with player info:
                - player_id, color, score, trains_remaining, cards_count, is_current
        """
        # Clear existing labels
        for i in reversed(range(self.players_layout.count())):
            self.players_layout.itemAt(i).widget().setParent(None)
        
        self.player_labels = {}
        
        for player_data in players_data:
            player_id = player_data['player_id']
            
            # Create player info group
            group = QGroupBox(f"Player {player_id + 1} - {player_data.get('color', 'Unknown')}")
            if player_data.get('is_current', False):
                group.setStyleSheet("QGroupBox { font-weight: bold; border: 2px solid blue; }")
            
            grid = QGridLayout()
            
            # Score
            grid.addWidget(QLabel("Score:"), 0, 0)
            score_label = QLabel(str(player_data.get('score', 0)))
            grid.addWidget(score_label, 0, 1)
            self.player_labels[f"{player_id}_score"] = score_label
            
            # Trains remaining
            grid.addWidget(QLabel("Trains:"), 1, 0)
            trains_label = QLabel(str(player_data.get('trains_remaining', 0)))
            grid.addWidget(trains_label, 1, 1)
            self.player_labels[f"{player_id}_trains"] = trains_label
            
            # Cards count
            grid.addWidget(QLabel("Cards:"), 2, 0)
            cards_label = QLabel(str(player_data.get('cards_count', 0)))
            grid.addWidget(cards_label, 2, 1)
            self.player_labels[f"{player_id}_cards"] = cards_label
            
            group.setLayout(grid)
            self.players_layout.addWidget(group)

