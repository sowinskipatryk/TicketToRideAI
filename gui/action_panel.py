"""Action panel widget."""
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QPushButton, QGroupBox, QVBoxLayout
from PyQt6.QtCore import pyqtSignal


class ActionPanel(QWidget):
    """Panel with action buttons."""
    
    action_requested = pyqtSignal(str)  # Emits action name
    
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI."""
        layout = QVBoxLayout(self)
        
        group = QGroupBox("Actions")
        group_layout = QHBoxLayout()
        
        # Action buttons
        self.claim_route_btn = QPushButton("Claim Route")
        self.claim_route_btn.clicked.connect(lambda: self.action_requested.emit("claim_route"))
        group_layout.addWidget(self.claim_route_btn)
        
        self.draw_cards_btn = QPushButton("Draw Cards")
        self.draw_cards_btn.clicked.connect(lambda: self.action_requested.emit("draw_cards"))
        group_layout.addWidget(self.draw_cards_btn)
        
        self.draw_tickets_btn = QPushButton("Draw Tickets")
        self.draw_tickets_btn.clicked.connect(lambda: self.action_requested.emit("draw_tickets"))
        group_layout.addWidget(self.draw_tickets_btn)
        
        self.skip_btn = QPushButton("Skip")
        self.skip_btn.clicked.connect(lambda: self.action_requested.emit("skip"))
        group_layout.addWidget(self.skip_btn)
        
        group.setLayout(group_layout)
        layout.addWidget(group)
    
    def set_actions_enabled(self, claim_route: bool, draw_cards: bool,
                           draw_tickets: bool, skip: bool):
        """Enable/disable action buttons."""
        self.claim_route_btn.setEnabled(claim_route)
        self.draw_cards_btn.setEnabled(draw_cards)
        self.draw_tickets_btn.setEnabled(draw_tickets)
        self.skip_btn.setEnabled(skip)

