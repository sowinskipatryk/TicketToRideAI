"""Dialog for selecting number of wild cards to use."""
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QSpinBox, QGroupBox)
from PyQt6.QtCore import Qt


class WildCardsDialog(QDialog):
    """Dialog for selecting number of wild cards to use."""
    
    def __init__(self, max_wild_cards: int, route_length: int, parent=None):
        """Initialize wild cards selection dialog.
        
        Args:
            max_wild_cards: Maximum number of wild cards available
            route_length: Length of the route being claimed
            parent: Parent widget
        """
        super().__init__(parent)
        self.max_wild_cards = max_wild_cards
        self.route_length = route_length
        self.selected_count = 0
        self.setWindowTitle("Select Wild Cards")
        self.setModal(True)
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI."""
        layout = QVBoxLayout(self)
        
        # Instructions
        info_label = QLabel(
            f"Route length: {self.route_length}\n"
            f"Available wild cards: {self.max_wild_cards}\n\n"
            f"Select how many wild cards to use (0-{min(self.max_wild_cards, self.route_length)}):"
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # Wild cards selection group
        wild_group = QGroupBox("Wild Cards")
        wild_layout = QVBoxLayout()
        
        self.spin_box = QSpinBox()
        self.spin_box.setMinimum(0)
        self.spin_box.setMaximum(min(self.max_wild_cards, self.route_length))
        self.spin_box.setValue(0)  # Default to 0
        self.spin_box.valueChanged.connect(self.on_value_changed)
        wild_layout.addWidget(self.spin_box)
        
        wild_group.setLayout(wild_layout)
        layout.addWidget(wild_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        button_layout.addWidget(ok_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
    
    def on_value_changed(self, value: int):
        """Handle value change."""
        self.selected_count = value
    
    def get_wild_cards_count(self) -> int:
        """Get the selected wild cards count.
        
        Returns:
            Number of wild cards to use, or 0 if cancelled
        """
        if self.exec() == QDialog.DialogCode.Accepted:
            return self.spin_box.value()
        return 0  # Default to 0 if cancelled

