"""Main GUI window for Ticket to Ride game."""
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QMessageBox, QSplitter,
                             QTextEdit, QGroupBox, QGridLayout)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont

from gui.game_board_widget import GameBoardWidget
from gui.player_info_widget import PlayerInfoWidget
from gui.card_widget import CardWidget
from gui.action_panel import ActionPanel
from gui.game_controller_gui import GameControllerGUI


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.game_controller_gui = None
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI."""
        self.setWindowTitle("Ticket to Ride AI")
        self.setGeometry(100, 100, 1600, 1000)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel - Game board
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Game board widget
        self.board_widget = GameBoardWidget()
        left_layout.addWidget(self.board_widget)
        
        # Action panel
        self.action_panel = ActionPanel()
        self.action_panel.action_requested.connect(self.handle_action)
        left_layout.addWidget(self.action_panel)
        
        splitter.addWidget(left_panel)
        
        # Right panel - Info and cards
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Player info widget
        self.player_info = PlayerInfoWidget()
        right_layout.addWidget(self.player_info)
        
        # Cards widget
        self.card_widget = CardWidget()
        right_layout.addWidget(self.card_widget)
        
        # Game log
        log_group = QGroupBox("Game Log")
        log_layout = QVBoxLayout()
        self.game_log = QTextEdit()
        self.game_log.setReadOnly(True)
        self.game_log.setMaximumHeight(200)
        log_layout.addWidget(self.game_log)
        log_group.setLayout(log_layout)
        right_layout.addWidget(log_group)
        
        splitter.addWidget(right_panel)
        
        # Set splitter proportions (65% board, 35% info) - give more space to board
        splitter.setSizes([1040, 560])
        
        # Menu bar
        self.create_menu_bar()
        
        # Status bar
        self.statusBar().showMessage("Ready")
    
    def create_menu_bar(self):
        """Create menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        new_game_action = file_menu.addAction("New Game")
        new_game_action.triggered.connect(self.new_game)
        
        file_menu.addSeparator()
        
        exit_action = file_menu.addAction("Exit")
        exit_action.triggered.connect(self.close)
        
        # Game menu
        game_menu = menubar.addMenu("Game")
        
        pause_action = game_menu.addAction("Pause")
        pause_action.triggered.connect(self.pause_game)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        about_action = help_menu.addAction("About")
        about_action.triggered.connect(self.show_about)
    
    def new_game(self):
        """Start a new game."""
        from gui.new_game_dialog import NewGameDialog
        
        dialog = NewGameDialog(self)
        if dialog.exec():
            player_types, version = dialog.get_settings()
            
            # Create game controller GUI
            self.game_controller_gui = GameControllerGUI(
                player_types=player_types,
                version=version,
                board_widget=self.board_widget,
                player_info=self.player_info,
                card_widget=self.card_widget,
                action_panel=self.action_panel,
                log_widget=self.game_log,
                parent_window=self  # Pass main window as parent for dialogs
            )
            
            # Connect signals
            self.game_controller_gui.status_message.connect(
                self.statusBar().showMessage
            )
            
            # Start the game
            self.game_controller_gui.start_game()
            self.statusBar().showMessage("Game started")
    
    def handle_action(self, action: str):
        """Handle action from action panel."""
        if self.game_controller_gui:
            self.game_controller_gui.handle_player_action(action)
    
    def pause_game(self):
        """Pause/resume the game."""
        if self.game_controller_gui:
            self.game_controller_gui.toggle_pause()
    
    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About Ticket to Ride AI",
            "Ticket to Ride AI\n\n"
            "A Python implementation of Ticket to Ride\n"
            "with AI players using NEAT neural networks.\n\n"
            "Version 1.0"
        )
    
    def closeEvent(self, event):
        """Handle window close event."""
        if self.game_controller_gui:
            reply = QMessageBox.question(
                self,
                "Quit",
                "Are you sure you want to quit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

