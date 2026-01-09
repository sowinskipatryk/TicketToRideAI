"""Human player that uses CLI or GUI based on context."""
from game.players.base_player import BasePlayer
from typing import List, Optional
from game.ticket_deck import Ticket


class HumanPlayer(BasePlayer):
    """Human player that can work with CLI or GUI.
    
    When used in GUI mode, decision methods will be called by the GUI controller.
    When used in CLI mode, it uses CLI interface.
    """
    
    def __init__(self, color_index: int, game: 'Game', adapter, ui_interface=None):
        """Initialize human player.
        
        Args:
            color_index: Player color index
            game: Game instance
            adapter: Network adapter
            ui_interface: Optional UI interface (CLI or GUI). If None, detects automatically.
        """
        super().__init__(color_index, game, adapter)
        self.ui_interface = ui_interface
        self._pending_ticket_decision = None  # For GUI ticket selection
    
    def decide_tickets(self, min_keep: int, tickets: List[Ticket]):
        """Get ticket selection from human player.
        
        Args:
            min_keep: Minimum number of tickets to keep
            tickets: List of tickets to choose from
            
        Returns:
            Tuple[List[int], List[int]]: (kept_ticket_indices, discarded_ticket_indices)
        """
        # If GUI interface is set, wait for GUI decision
        if self.ui_interface and hasattr(self.ui_interface, 'request_ticket_selection'):
            return self.ui_interface.request_ticket_selection(self, min_keep, tickets)
        
        # Otherwise use CLI
        try:
            from cli.human_player_cli import HumanPlayerCLI
            return HumanPlayerCLI.decide_tickets(min_keep, tickets, self.game)
        except ImportError:
            # Fallback: keep minimum required tickets
            return list(range(min_keep)), list(range(min_keep, len(tickets)))
    
    def decide_action(self) -> int:
        """Get action decision from human player."""
        # GUI handles this through action panel, CLI uses input
        if self.ui_interface and hasattr(self.ui_interface, 'request_action'):
            return self.ui_interface.request_action(self)
        
        try:
            from cli.human_player_cli import HumanPlayerCLI
            return HumanPlayerCLI.decide_action(self.game)
        except ImportError:
            return 3  # Skip as fallback
    
    def decide_wild_cards(self) -> int:
        """Get wild card count from human player."""
        if self.ui_interface and hasattr(self.ui_interface, 'request_wild_cards'):
            return self.ui_interface.request_wild_cards(self)
        
        try:
            from cli.human_player_cli import HumanPlayerCLI
            return HumanPlayerCLI.decide_wild_cards(self.game)
        except ImportError:
            return 0
    
    def decide_cards_color(self) -> int:
        """Get card color decision from human player."""
        if self.ui_interface and hasattr(self.ui_interface, 'request_card_color'):
            return self.ui_interface.request_card_color(self)
        
        try:
            from cli.human_player_cli import HumanPlayerCLI
            return HumanPlayerCLI.decide_cards_color(self.hand, self.game)
        except ImportError:
            return 0
    
    def decide_train_card(self) -> int:
        """Get train card selection from human player."""
        if self.ui_interface and hasattr(self.ui_interface, 'request_train_card'):
            return self.ui_interface.request_train_card(self)
        
        try:
            from cli.human_player_cli import HumanPlayerCLI
            return HumanPlayerCLI.decide_train_card(self.game)
        except ImportError:
            return 5  # Draw pile as fallback
    
    def decide_route(self) -> int:
        """Get route selection from human player."""
        if self.ui_interface and hasattr(self.ui_interface, 'request_route'):
            return self.ui_interface.request_route(self)
        
        try:
            from cli.human_player_cli import HumanPlayerCLI
            return HumanPlayerCLI.decide_route(self.game)
        except ImportError:
            return 0
