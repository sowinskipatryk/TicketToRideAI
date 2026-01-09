"""Dialog for starting a new game."""
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QComboBox, QSpinBox, QPushButton, QGroupBox,
                             QFormLayout, QMessageBox, QWidget)
from PyQt6.QtCore import Qt


class NewGameDialog(QDialog):
    """Dialog for configuring a new game."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("New Game")
        self.setModal(True)
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI."""
        layout = QVBoxLayout(self)
        
        # Game version
        version_group = QGroupBox("Game Version")
        version_layout = QFormLayout()
        
        self.version_combo = QComboBox()
        self.version_combo.addItems(["USA", "Europe", "Nordic"])
        version_layout.addRow("Version:", self.version_combo)
        
        version_group.setLayout(version_layout)
        layout.addWidget(version_group)
        
        # Players
        players_group = QGroupBox("Players")
        players_layout = QFormLayout()
        
        self.num_players_spin = QSpinBox()
        self.num_players_spin.setMinimum(2)
        self.num_players_spin.setMaximum(5)
        self.num_players_spin.setValue(2)
        self.num_players_spin.valueChanged.connect(self.update_player_types)
        players_layout.addRow("Number of Players:", self.num_players_spin)
        
        self.player_types = []
        self.player_type_layout = QVBoxLayout()
        players_layout.addRow("Player Types:", self.player_type_layout)
        
        players_group.setLayout(players_layout)
        layout.addWidget(players_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.ok_btn = QPushButton("Start Game")
        self.ok_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.ok_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
        
        # Initialize player types
        self.update_player_types()
    
    def update_player_types(self):
        """Update player type combo boxes."""
        num_players = self.num_players_spin.value()
        
        # Clear existing
        for i in reversed(range(self.player_type_layout.count())):
            item = self.player_type_layout.itemAt(i)
            if item.widget():
                item.widget().setParent(None)
        
        # Create new combo boxes
        self.player_combos = []
        for i in range(num_players):
            label = QLabel(f"Player {i + 1}:")
            combo = QComboBox()
            combo.addItems(["Human", "NEAT", "Random"])
            if i == 0:
                combo.setCurrentText("Human")  # First player is human by default
            
            h_layout = QHBoxLayout()
            h_layout.addWidget(label)
            h_layout.addWidget(combo)
            h_layout.addStretch()
            
            widget = QWidget()
            widget.setLayout(h_layout)
            self.player_type_layout.addWidget(widget)
            self.player_combos.append(combo)
    
    def get_settings(self):
        """Get game settings.
        
        Returns:
            Tuple of (player_types list, version string)
        """
        player_types = [combo.currentText() for combo in self.player_combos]
        version = self.version_combo.currentText()
        return player_types, version

