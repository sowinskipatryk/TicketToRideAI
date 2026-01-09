"""Card display widget."""
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QGroupBox, QPushButton, QScrollArea)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen, QBrush


class CardWidget(QWidget):
    """Widget for displaying train cards and tickets."""
    
    card_clicked = pyqtSignal(str)  # Emits card color when clicked
    
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI."""
        layout = QVBoxLayout(self)
        
        # Hand cards
        hand_group = QGroupBox("Your Hand")
        hand_layout = QVBoxLayout()
        
        self.hand_cards_layout = QHBoxLayout()
        hand_layout.addLayout(self.hand_cards_layout)
        hand_layout.addStretch()
        
        hand_group.setLayout(hand_layout)
        layout.addWidget(hand_group)
        
        # Face-up cards
        face_up_group = QGroupBox("Face-Up Cards")
        face_up_layout = QVBoxLayout()
        
        self.face_up_layout = QHBoxLayout()
        face_up_layout.addLayout(self.face_up_layout)
        
        face_up_group.setLayout(face_up_layout)
        layout.addWidget(face_up_group)
        
        # Tickets
        tickets_group = QGroupBox("Your Tickets")
        tickets_layout = QVBoxLayout()
        
        self.tickets_area = QScrollArea()
        self.tickets_widget = QWidget()
        self.tickets_layout = QVBoxLayout(self.tickets_widget)
        self.tickets_area.setWidget(self.tickets_widget)
        self.tickets_area.setWidgetResizable(True)
        self.tickets_area.setMaximumHeight(200)
        
        tickets_layout.addWidget(self.tickets_area)
        tickets_group.setLayout(tickets_layout)
        layout.addWidget(tickets_group)
    
    def update_hand(self, hand: dict):
        """Update hand display.
        
        Args:
            hand: Dict mapping color to count
        """
        # Clear existing cards
        for i in reversed(range(self.hand_cards_layout.count())):
            item = self.hand_cards_layout.itemAt(i)
            if item.widget():
                item.widget().setParent(None)
        
        # Add cards
        for color, count in sorted(hand.items()):
            if count > 0:
                card = CardButton(color, count)
                card.clicked.connect(lambda checked, c=color: self.card_clicked.emit(c))
                self.hand_cards_layout.addWidget(card)
        
        self.hand_cards_layout.addStretch()
    
    def update_face_up_cards(self, cards: list):
        """Update face-up cards display.
        
        Args:
            cards: List of card colors (or None)
        """
        # Clear existing
        for i in reversed(range(self.face_up_layout.count())):
            item = self.face_up_layout.itemAt(i)
            if item.widget():
                item.widget().setParent(None)
        
        # Add cards
        for i, card in enumerate(cards):
            if card:
                card_btn = CardButton(card, 1, clickable=True, index=i)
                card_btn.clicked.connect(lambda checked, idx=i: self.card_clicked.emit(f"face_up_{idx}"))
                self.face_up_layout.addWidget(card_btn)
            else:
                empty_label = QLabel("Empty")
                empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                empty_label.setStyleSheet("border: 1px solid gray; padding: 10px;")
                self.face_up_layout.addWidget(empty_label)
        
        # Draw pile button
        draw_btn = QPushButton("Draw\nPile")
        draw_btn.clicked.connect(lambda: self.card_clicked.emit("draw_pile"))
        self.face_up_layout.addWidget(draw_btn)
        
        self.face_up_layout.addStretch()
    
    def update_tickets(self, tickets: dict):
        """Update tickets display.
        
        Args:
            tickets: Dict mapping Ticket to completed (bool)
        """
        # Clear existing
        for i in reversed(range(self.tickets_layout.count())):
            item = self.tickets_layout.itemAt(i)
            if item.widget():
                item.widget().setParent(None)
        
        # Add tickets
        for ticket, completed in tickets.items():
            status = "✓" if completed else "✗"
            color = "green" if completed else "red"
            label = QLabel(
                f"{status} {ticket.city_from} → {ticket.city_to} ({ticket.points} pts)"
            )
            label.setStyleSheet(f"color: {color}; padding: 5px;")
            self.tickets_layout.addWidget(label)
        
        self.tickets_layout.addStretch()


class CardButton(QPushButton):
    """Button representing a train card."""
    
    def __init__(self, color: str, count: int = 1, clickable: bool = False, index: int = -1):
        super().__init__()
        self.color = color
        self.count = count
        self.clickable = clickable
        
        if count > 1:
            self.setText(f"{color}\n×{count}")
        else:
            self.setText(color)
        
        self.setMinimumSize(60, 80)
        self.setMaximumSize(60, 80)
        
        # Set color-based styling
        color_map = {
            'red': '#FF0000',
            'blue': '#0000FF',
            'green': '#00FF00',
            'yellow': '#FFFF00',
            'black': '#000000',
            'white': '#FFFFFF',
            'pink': '#FFC0CB',
            'orange': '#FFA500',
            'wild': '#800080',
        }
        
        bg_color = color_map.get(color.lower(), '#CCCCCC')
        text_color = '#FFFFFF' if color.lower() in ['black', 'blue'] else '#000000'
        
        if color.lower() == 'wild':
            self.setStyleSheet(
                f"background-color: {bg_color}; color: {text_color}; "
                f"border: 2px solid gold; font-weight: bold;"
            )
        else:
            self.setStyleSheet(
                f"background-color: {bg_color}; color: {text_color}; "
                f"border: 1px solid black;"
            )
        
        if not clickable:
            self.setEnabled(False)

