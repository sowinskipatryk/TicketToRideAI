"""Dialog for selecting card color for grey routes."""
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QButtonGroup, QRadioButton, QGroupBox)
from PyQt6.QtCore import Qt
from typing import List


class CardColorDialog(QDialog):
    """Dialog for selecting card color for grey routes."""
    
    def __init__(self, available_colors: List[str], parent=None):
        """Initialize card color selection dialog.
        
        Args:
            available_colors: List of available train colors
            parent: Parent widget
        """
        super().__init__(parent)
        self.available_colors = available_colors
        self.selected_color_index = None
        self.setWindowTitle("Select Card Color")
        self.setModal(True)
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI."""
        layout = QVBoxLayout(self)
        
        # Instructions
        info_label = QLabel("This route requires a specific color. Select which color to use:")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # Color selection group
        color_group = QGroupBox("Card Color")
        color_layout = QVBoxLayout()
        
        self.button_group = QButtonGroup(self)
        self.radio_buttons = []
        
        for i, color in enumerate(self.available_colors):
            radio = QRadioButton(color.capitalize())
            radio.setChecked(i == 0)  # Select first by default
            self.radio_buttons.append(radio)
            self.button_group.addButton(radio, i)
            color_layout.addWidget(radio)
        
        color_group.setLayout(color_layout)
        layout.addWidget(color_group)
        
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
    
    def get_selected_color_index(self) -> int:
        """Get the selected color index.
        
        Returns:
            Index of selected color, or 0 if cancelled
        """
        if self.exec() == QDialog.DialogCode.Accepted:
            checked_button = self.button_group.checkedButton()
            if checked_button:
                return self.button_group.id(checked_button)
        return 0  # Default to first color if cancelled

