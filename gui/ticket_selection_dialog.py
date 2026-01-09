"""Dialog for selecting tickets."""
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QCheckBox, QGroupBox, QMessageBox)
from PyQt6.QtCore import Qt, pyqtSignal
from typing import List
from game.ticket_deck import Ticket


class TicketSelectionDialog(QDialog):
    """Dialog for selecting which tickets to keep."""
    
    def __init__(self, tickets: List[Ticket], min_keep: int, parent=None):
        """Initialize ticket selection dialog.
        
        Args:
            tickets: List of tickets to choose from
            min_keep: Minimum number of tickets that must be kept
            parent: Parent widget
        """
        super().__init__(parent)
        self.tickets = tickets
        self.min_keep = min_keep
        self.checkboxes = []
        self._selection_cache = None  # Cache selection to avoid accessing after destruction
        self.setWindowTitle("Select Tickets")
        self.setModal(True)
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI."""
        layout = QVBoxLayout(self)
        
        # Instructions
        info_label = QLabel(
            f"Select at least {self.min_keep} ticket(s) to keep. "
            f"Unselected tickets will be discarded."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # Tickets group
        tickets_group = QGroupBox("Tickets")
        tickets_layout = QVBoxLayout()
        
        self.checkboxes = []
        for i, ticket in enumerate(self.tickets):
            checkbox = QCheckBox(
                f"{ticket.city_from} → {ticket.city_to} ({ticket.points} points)"
            )
            checkbox.setChecked(True)  # Default: keep all
            self.checkboxes.append(checkbox)
            tickets_layout.addWidget(checkbox)
        
        tickets_group.setLayout(tickets_layout)
        layout.addWidget(tickets_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        # Select all / Deselect all buttons
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self.select_all)
        button_layout.addWidget(select_all_btn)
        
        deselect_all_btn = QPushButton("Deselect All")
        deselect_all_btn.clicked.connect(self.deselect_all)
        button_layout.addWidget(deselect_all_btn)
        
        button_layout.addStretch()
        
        # OK and Cancel buttons
        self.ok_btn = QPushButton("OK")
        self.ok_btn.clicked.connect(self.accept_selection)
        button_layout.addWidget(self.ok_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
        
        # Update OK button state
        self.update_ok_button()
        
        # Connect checkbox changes to update OK button
        for checkbox in self.checkboxes:
            checkbox.stateChanged.connect(self.update_ok_button)
    
    def select_all(self):
        """Select all tickets."""
        for checkbox in self.checkboxes:
            checkbox.setChecked(True)
    
    def deselect_all(self):
        """Deselect all tickets."""
        for checkbox in self.checkboxes:
            checkbox.setChecked(False)
    
    def update_ok_button(self):
        """Update OK button enabled state based on selection."""
        if not hasattr(self, 'ok_btn') or not self.checkboxes:
            return
        try:
            selected_count = sum(1 for cb in self.checkboxes if cb.isChecked())
            self.ok_btn.setEnabled(selected_count >= self.min_keep)
        except (AttributeError, RuntimeError):
            # Dialog might be closing, ignore
            pass
    
    def accept_selection(self):
        """Accept the selection if valid."""
        try:
            if not hasattr(self, 'checkboxes') or not self.checkboxes:
                self._selection_cache = ([], [])
                self.accept()
                return
            selected_count = sum(1 for cb in self.checkboxes if cb.isChecked())
            if selected_count < self.min_keep:
                QMessageBox.warning(
                    self,
                    "Invalid Selection",
                    f"You must select at least {self.min_keep} ticket(s)."
                )
                return
            
            # Cache selection before accepting
            self._selection_cache = self._get_selection_internal()
            self.accept()
        except (AttributeError, RuntimeError) as e:
            # Dialog might be closing, just accept to avoid crash
            try:
                if self._selection_cache is None:
                    self._selection_cache = ([], [])
                self.accept()
            except:
                pass
    
    def _get_selection_internal(self) -> tuple:
        """Internal method to get selection without error handling."""
        kept_indices = [i for i, cb in enumerate(self.checkboxes) if cb.isChecked()]
        discarded_indices = [i for i, cb in enumerate(self.checkboxes) if not cb.isChecked()]
        return kept_indices, discarded_indices
    
    def get_selection(self) -> tuple:
        """Get the selected ticket indices.
        
        Returns:
            Tuple[List[int], List[int]]: (kept_ticket_indices, discarded_ticket_indices)
        """
        # Use cached selection if available (dialog might be destroyed)
        if self._selection_cache is not None:
            return self._selection_cache
        
        if not hasattr(self, 'checkboxes') or not self.checkboxes:
            # Return empty selection if checkboxes not available
            return [], []
        try:
            return self._get_selection_internal()
        except (AttributeError, RuntimeError):
            # Dialog might be closing, return cached or empty
            return self._selection_cache if self._selection_cache is not None else ([], [])

