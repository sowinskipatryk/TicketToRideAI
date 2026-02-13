"""GUI controller that connects game logic to GUI."""
import time
from PyQt6.QtCore import QObject, pyqtSignal, QTimer
from PyQt6.QtWidgets import QApplication
from typing import List, Optional

from game.game_controller import GameController
from game.events import EventType, GameEvent
from game.enums import ActionDecision, GameState
from gui.game_board_widget import GameBoardWidget
from gui.player_info_widget import PlayerInfoWidget
from gui.card_widget import CardWidget
from gui.action_panel import ActionPanel
from gui.ticket_selection_dialog import TicketSelectionDialog
from gui.card_color_dialog import CardColorDialog
from gui.wild_cards_dialog import WildCardsDialog


class GameControllerGUI(QObject):
    """Controller that manages game state and updates GUI."""
    
    status_message = pyqtSignal(str)
    
    def __init__(self, player_types: List[str], version: str,
                 board_widget: GameBoardWidget,
                 player_info: PlayerInfoWidget,
                 card_widget: CardWidget,
                 action_panel: ActionPanel,
                 log_widget,
                 parent_window=None):
        super().__init__()
        
        self.board_widget = board_widget
        self.player_info = player_info
        self.card_widget = card_widget
        self.action_panel = action_panel
        self.log_widget = log_widget
        self.parent_window = parent_window  # Store parent window for dialogs
        
        # Card drawing state tracking
        self.cards_drawn_this_turn = 0
        self.wild_card_drawn = False
        self.normal_card_drawn = False
        self.is_drawing_cards = False
        
        # Create game controller with GUI interface for human players
        self.game_controller = GameController(player_types, version, ui_interface=self)
        
        # Connect board widget signals
        self.board_widget.route_clicked.connect(self.handle_route_click)
        
        # Connect card widget signals
        self.card_widget.card_clicked.connect(self.handle_card_click)
        
        # Subscribe to game events
        self.game_controller.event_bus.subscribe(EventType.GAME_STARTED, self.on_game_started)
        self.game_controller.event_bus.subscribe(EventType.TURN_STARTED, self.on_turn_started)
        self.game_controller.event_bus.subscribe(EventType.TURN_ENDED, self.on_turn_ended)
        self.game_controller.event_bus.subscribe(EventType.ROUTE_CLAIMED, self.on_route_claimed)
        self.game_controller.event_bus.subscribe(EventType.GAME_ENDED, self.on_game_ended)
        
        self.paused = False
        self.ai_timer = QTimer()
        self.ai_timer.timeout.connect(self.process_ai_turn)
        
        # State for current action
        self.current_action = None
        self.pending_route_id = None
        
        # State for pending UI decisions (used by HumanPlayer UI interface methods)
        self._pending_route_decision = None
        self._pending_card_color_decision = None
        self._pending_wild_cards_decision = None
        self._waiting_for_route = False
        self._waiting_for_card_color = False
        self._waiting_for_wild_cards = False
        
        # Initial setup state
        self.initial_setup_done = False
        self.current_setup_player = 0
    
    def start_game(self):
        """Start the game with initial setup."""
        game = self.game_controller.game

        # Initialize board with geographic coordinates if available
        self.board_widget.set_graph(
            game.board.G,
            game.board.cities,
            city_coordinates=game.board.city_coordinates
        )

        # Disable actions during initial setup
        self.action_panel.set_actions_enabled(False, False, False, False)
        
        # Start initial setup
        self.status_message.emit("Initial setup: Drawing cards and tickets...")
        self.perform_initial_setup()
    
    def perform_initial_setup(self):
        """Perform initial game setup: draw cards and tickets for all players."""
        game = self.game_controller.game
        
        # Set up ticket deck adapter
        game.ticket_deck.set_ticket_pile_num_adapter()
        
        # Process initial setup for each player
        self.setup_next_player()
    
    def setup_next_player(self):
        """Setup next player's initial cards and tickets."""
        game = self.game_controller.game
        
        if self.current_setup_player >= len(game.players):
            # All players setup complete
            try:
                self.initial_setup_done = True
                game.game_state = GameState.RUNNING
                self.status_message.emit("Initial setup complete! Game starting...")
                self.update_display()
                
                # Enable actions for current player
                is_human = self.is_current_player_human()
                self.action_panel.set_actions_enabled(
                    claim_route=is_human,
                    draw_cards=is_human,
                    draw_tickets=is_human,
                    skip=is_human
                )
                
                # Emit turn started event for first player
                current_player = game.get_current_player()
                self.game_controller.event_bus.emit(GameEvent(
                    EventType.TURN_STARTED,
                    player_id=current_player.player_id,
                    data={'move_number': self.game_controller._current_move},
                    timestamp=time.time()
                ))
                
                # Start first turn if AI player
                if not is_human:
                    self.ai_timer.start(1000)
                else:
                    self.status_message.emit("Your turn")
            except Exception as e:
                self.log_message(f"Error completing setup: {e}")
                import traceback
                self.log_message(traceback.format_exc())
            return
        
        player = game.players[self.current_setup_player]
        self.status_message.emit(f"Setting up Player {self.current_setup_player + 1} ({player.color})...")
        
        # Draw initial train cards
        try:
            player.draw_initial_train_cards(game.config.NUM_TRAIN_CARDS_DEALT_INIT)
            self.log_message(f"Player {self.current_setup_player + 1} drew {game.config.NUM_TRAIN_CARDS_DEALT_INIT} train cards")
        except Exception as e:
            self.log_message(f"Error drawing initial cards for player {self.current_setup_player + 1}: {e}")
            import traceback
            self.log_message(traceback.format_exc())
            # Continue anyway to avoid blocking
        
        # Draw initial tickets
        tickets = game.deal_tickets(game.config.NUM_TICKETS_DEALT_INIT)
        if tickets:
            # Check if this is a human player
            if player.__class__.__name__ == 'HumanPlayer':
                # Show ticket selection dialog for human player
                dialog = None
                kept_indices = []
                discarded_indices = []
                
                try:
                    dialog = TicketSelectionDialog(
                        tickets,
                        game.config.MIN_TICKETS_KEPT_INIT,
                        parent=self.parent_window
                    )
                    
                    result = dialog.exec()
                    
                    # Get selection immediately after exec() returns and store locally
                    # This prevents accessing dialog after it might be destroyed
                    if result:
                        try:
                            kept_indices, discarded_indices = dialog.get_selection()
                        except Exception as e:
                            self.log_message(f"Error getting selection: {e}")
                            kept_indices = []
                            discarded_indices = []
                        
                        # Validate indices
                        kept_indices = [idx for idx in kept_indices if 0 <= idx < len(tickets)]
                        discarded_indices = [idx for idx in discarded_indices if 0 <= idx < len(tickets)]
                        
                        # Ensure minimum kept
                        if len(kept_indices) < game.config.MIN_TICKETS_KEPT_INIT:
                            # Fallback: keep first min_keep tickets
                            kept_indices = list(range(game.config.MIN_TICKETS_KEPT_INIT))
                            discarded_indices = [i for i in range(len(tickets)) if i not in kept_indices]
                    else:
                        # User cancelled - keep minimum required
                        kept_indices = list(range(min(game.config.MIN_TICKETS_KEPT_INIT, len(tickets))))
                        discarded_indices = [i for i in range(len(tickets)) if i not in kept_indices]
                except Exception as e:
                    # Error handling - fallback to minimum keep
                    self.log_message(f"Error in ticket selection dialog: {e}")
                    import traceback
                    self.log_message(traceback.format_exc())
                    kept_indices = list(range(min(game.config.MIN_TICKETS_KEPT_INIT, len(tickets))))
                    discarded_indices = [i for i in range(len(tickets)) if i not in kept_indices]
                finally:
                    # Ensure dialog is cleaned up
                    if dialog is not None:
                        try:
                            dialog.close()
                            dialog.deleteLater()
                        except:
                            pass
                        dialog = None
                
                # Now process the tickets using the stored indices
                # This happens after dialog is closed/destroyed
                try:
                    for idx in kept_indices:
                        if idx < len(tickets):
                            player.add_ticket(tickets[idx])
                            self.log_message(f"Player {self.current_setup_player + 1} kept: {tickets[idx].city_from} → {tickets[idx].city_to} ({tickets[idx].points} pts)")
                    
                    # Return discarded tickets to deck
                    for idx in discarded_indices:
                        if idx < len(tickets):
                            game.ticket_deck.insert(tickets[idx])
                except Exception as e:
                    self.log_message(f"Error processing tickets: {e}")
                    # Fallback: keep minimum
                    for i in range(min(game.config.MIN_TICKETS_KEPT_INIT, len(tickets))):
                        try:
                            player.add_ticket(tickets[i])
                        except:
                            pass
            else:
                # AI or random player - use their decision method
                kept, discarded = player.decide_tickets(game.config.MIN_TICKETS_KEPT_INIT, tickets)
                
                for idx in kept:
                    player.add_ticket(tickets[idx])
                    self.log_message(f"Player {self.current_setup_player + 1} kept: {tickets[idx].city_from} → {tickets[idx].city_to}")
                
                for idx in discarded:
                    game.ticket_deck.insert(tickets[idx])
        
        # Update display
        self.update_display()
        
        # Move to next player
        self.current_setup_player += 1
        
        # Process next player setup (with small delay for GUI responsiveness)
        # Use QApplication.processEvents() to ensure GUI updates
        try:
            QApplication.processEvents()
        except:
            pass
        
        if self.current_setup_player < len(game.players):
            # Schedule next player setup with delay
            # Use a method reference instead of lambda to avoid closure issues
            # Add safety check in callback
            def safe_setup_next():
                try:
                    if hasattr(self, 'current_setup_player') and hasattr(self, 'game_controller'):
                        self.setup_next_player()
                except Exception as e:
                    self.log_message(f"Error in setup: {e}")
            QTimer.singleShot(500, safe_setup_next)  # 500ms delay
        else:
            # Complete setup immediately
            try:
                self.setup_next_player()
            except Exception as e:
                self.log_message(f"Error completing setup: {e}")
                # Force completion if error
                self.initial_setup_done = True
                game.game_state = GameState.RUNNING
                self.update_display()
    
    def update_display(self):
        """Update all GUI elements."""
        game = self.game_controller.game
        current_player = game.get_current_player() if self.initial_setup_done else None
        
        # Update board
        claimed_routes = {}
        for u, v, data in game.board.G.edges(data=True):
            link_id = data.get('link_id')
            if link_id is not None:
                claimed_by = data.get('claimed_by')
                if claimed_by:
                    claimed_routes[link_id] = claimed_by
        
        self.board_widget.set_claimed_routes(claimed_routes)
        
        # Update available routes (simplified - would need validation logic)
        if self.initial_setup_done:
            available_routes = set(range(game.board.get_route_links_num()))
            self.board_widget.set_available_routes(available_routes)
        else:
            self.board_widget.set_available_routes(set())  # No routes available during setup
        
        # Update player info
        players_data = []
        for i, player in enumerate(game.players):
            players_data.append({
                'player_id': i,
                'color': str(player.color),
                'score': player.score,
                'trains_remaining': player.trains_remaining,
                'cards_count': sum(player.hand.values()),
                'is_current': i == game.current_player_id if self.initial_setup_done else False
            })
        self.player_info.update_players(players_data)
        
        # Update cards (show current player or first player during setup)
        display_player = current_player if current_player else (game.players[0] if game.players else None)
        if display_player:
            self.card_widget.update_hand(display_player.hand)
            self.card_widget.update_face_up_cards(
                game.train_card_manager.get_face_up_cards()
            )
            self.card_widget.update_tickets(display_player.tickets)
        
        # Update action panel (only if setup is done)
        if self.initial_setup_done:
            is_human = self.is_current_player_human()
            # Disable other actions if drawing cards
            if self.is_drawing_cards:
                self.action_panel.set_actions_enabled(
                    claim_route=False,
                    draw_cards=False,
                    draw_tickets=False,
                    skip=is_human  # Allow skip to end card drawing
                )
            else:
                self.action_panel.set_actions_enabled(
                    claim_route=is_human,
                    draw_cards=is_human,
                    draw_tickets=is_human,
                    skip=is_human
                )
        
        # Update card selection state
        self.update_card_selection_state()
    
    def update_card_selection_state(self):
        """Update which cards can be clicked based on drawing state."""
        # Card selection state is handled in handle_card_click by checking rules
        # This method is called for consistency but validation happens in the click handler
        pass
    
    def is_current_player_human(self) -> bool:
        """Check if current player is human."""
        if not self.initial_setup_done:
            return False
        try:
            game = self.game_controller.game
            current_player = game.get_current_player()
            return current_player.__class__.__name__ == 'HumanPlayer'
        except Exception:
            return False
    
    def handle_player_action(self, action: str):
        """Handle action from action panel."""
        # Block actions during initial setup
        if not self.initial_setup_done:
            self.status_message.emit("Please wait for initial setup to complete")
            return
        
        if not self.is_current_player_human():
            return
        
        game = self.game_controller.game
        current_player = game.get_current_player()
        
        if action == "claim_route":
            self.current_action = ActionDecision.CLAIM_ROUTE
            self.status_message.emit("Click on a route to claim it")
        elif action == "draw_cards":
            # Start card drawing mode
            self.is_drawing_cards = True
            self.cards_drawn_this_turn = 0
            self.wild_card_drawn = False
            self.normal_card_drawn = False
            self.status_message.emit("Click on a face-up card or draw pile to draw")
            # Enable card clicking
            self.update_card_selection_state()
        elif action == "draw_tickets":
            self.handle_draw_tickets()
        elif action == "skip":
            self.end_turn()
    
    def handle_route_click(self, link_id: int):
        """Handle route click."""
        # Block during initial setup
        if not self.initial_setup_done:
            return
        
        if self.current_action == ActionDecision.CLAIM_ROUTE:
            # Store the route selection for the UI interface method
            # This will be returned when request_route() is called
            self.pending_route_id = link_id
            self._pending_route_decision = link_id
            
            # Use the player's claim_route() method which handles all the game logic
            # This will call request_route(), request_card_color() (if grey), and request_wild_cards()
            game = self.game_controller.game
            current_player = game.get_current_player()
            
            # Call the player's claim_route method - it will call our UI interface methods
            # for route selection, card color (if grey), and wild cards
            move_completed = current_player.claim_route()
            
            # Clear pending route after use
            self.pending_route_id = None
            self._pending_route_decision = None
            
            if move_completed:
                city1, city2, route_data = game.board.get_route_data(link_id)
                self.log_message(f"Claimed route {city1}-{city2}")
                
                # Emit route claimed event
                self.game_controller.event_bus.emit(GameEvent(
                    EventType.ROUTE_CLAIMED,
                    player_id=current_player.player_id,
                    data={'route_id': link_id, 'cities': (city1, city2)},
                    timestamp=time.time()
                ))
                
                # Reset action state
                self.current_action = None
                
                self.update_display()
                self.end_turn()
            else:
                self.status_message.emit("Cannot claim this route - check if you have enough cards and trains")
    
    def handle_draw_tickets(self):
        """Handle draw tickets action."""
        game = self.game_controller.game
        current_player = game.get_current_player()
        
        # Use the player's draw_tickets method which handles all the game logic
        # This will call request_ticket_selection() for human players
        move_completed = current_player.draw_tickets(
            game.config.NUM_TICKETS_DEALT,
            game.config.MIN_TICKETS_KEPT
        )
        
        if move_completed:
            self.update_display()
            self.end_turn()
        else:
            self.status_message.emit("No tickets available or selection cancelled")
    
    def handle_card_click(self, card_id: str):
        """Handle card click."""
        # Only allow card clicks when in card drawing mode
        if not self.is_drawing_cards:
            return
        
        game = self.game_controller.game
        current_player = game.get_current_player()
        
        # Handle face-up card selection
        if card_id.startswith("face_up_"):
            index = int(card_id.split("_")[-1])
            face_up_cards = game.train_card_manager.get_face_up_cards()
            
            # Check if this is a wild card and if it's allowed
            if index < len(face_up_cards) and face_up_cards[index] == 'wild':
                # Wild card restrictions
                if game.config.WILD_CARD_RESTRICTION:
                    if self.normal_card_drawn:
                        # Cannot draw wild card after normal card
                        self.status_message.emit("Cannot draw wild card after drawing a normal card!")
                        return
                    if self.cards_drawn_this_turn > 0:
                        # Cannot draw wild card if already drew a card
                        self.status_message.emit("Cannot draw wild card after drawing another card!")
                        return
            
            card = game.deal_face_up_card(index)
            if card:
                current_player.add_cards_to_hand(card)
                self.cards_drawn_this_turn += 1
                
                if card == 'wild':
                    self.wild_card_drawn = True
                    self.log_message(f"Drew wild card")
                    # Wild card ends turn immediately (if restriction is enabled)
                    if game.config.WILD_CARD_RESTRICTION:
                        self.finish_card_drawing()
                        return
                else:
                    self.normal_card_drawn = True
                    self.log_message(f"Drew {card} card")
                
                # Check if we've drawn 2 cards (max per turn)
                if self.cards_drawn_this_turn >= 2:
                    self.finish_card_drawing()
                else:
                    self.update_display()
                    self.update_card_selection_state()
                    self.status_message.emit(f"Drew {card} card. Draw another card or end turn.")
        
        # Handle draw pile selection
        elif card_id == "draw_pile":
            card = game.deal_draw_pile_card()
            if card:
                current_player.add_cards_to_hand(card)
                self.cards_drawn_this_turn += 1
                self.normal_card_drawn = True
                self.log_message(f"Drew {card} card from deck")
                
                # Check if we've drawn 2 cards (max per turn)
                if self.cards_drawn_this_turn >= 2:
                    self.finish_card_drawing()
                else:
                    self.update_display()
                    self.update_card_selection_state()
                    self.status_message.emit(f"Drew {card} card. Draw another card or end turn.")
    
    def finish_card_drawing(self):
        """Finish the card drawing action and end turn."""
        self.is_drawing_cards = False
        self.cards_drawn_this_turn = 0
        self.wild_card_drawn = False
        self.normal_card_drawn = False
        self.update_display()
        self.update_card_selection_state()
        self.end_turn()
    
    def end_turn(self):
        """End current turn and advance to next player.
        
        This method is called after a human player manually completes an action.
        It uses the unified turn completion logic to handle state transitions
        and player advancement, ensuring consistency with CLI and headless modes.
        """
        # Block during initial setup
        if not self.initial_setup_done:
            return
        
        # If we're in card drawing mode, finish it first
        if self.is_drawing_cards:
            self.finish_card_drawing()
            return
        
        # Reset card drawing state
        self.is_drawing_cards = False
        self.cards_drawn_this_turn = 0
        self.wild_card_drawn = False
        self.normal_card_drawn = False
        
        self.current_action = None
        self.pending_route_id = None
        
        # Complete the turn using unified logic (handles last round, state transitions, etc.)
        # The turn was already played manually by the GUI, so we just complete it
        self.game_controller.complete_turn(move_completed=True)
        
        # Check if game is finished
        if self.game_controller.game.game_state == GameState.FINISHED:
            # Game ended - update display and return
            self.update_display()
            return
        
        # Check if next player is AI
        if not self.is_current_player_human():
            # Emit turn started event for AI player (since we're not calling play_turn yet)
            current_player = self.game_controller.game.get_current_player()
            self.game_controller.event_bus.emit(GameEvent(
                EventType.TURN_STARTED,
                player_id=current_player.player_id,
                data={'move_number': self.game_controller._current_move},
                timestamp=time.time()
            ))
            self.ai_timer.start(1000)
        else:
            # Emit turn started event for human player
            current_player = self.game_controller.game.get_current_player()
            self.game_controller.event_bus.emit(GameEvent(
                EventType.TURN_STARTED,
                player_id=current_player.player_id,
                data={'move_number': self.game_controller._current_move},
                timestamp=time.time()
            ))
            self.update_display()
    
    def process_ai_turn(self):
        """Process AI player turn."""
        self.ai_timer.stop()
        
        if self.paused:
            return
        
        self.status_message.emit("AI is thinking...")
        self.game_controller.play_turn()
        self.update_display()
        
        # Continue if still AI turn
        if not self.is_current_player_human():
            self.ai_timer.start(1000)
        else:
            self.status_message.emit("Your turn")
    
    def toggle_pause(self):
        """Toggle game pause."""
        self.paused = not self.paused
        if self.paused:
            self.ai_timer.stop()
            self.status_message.emit("Game paused")
        else:
            if not self.is_current_player_human():
                self.ai_timer.start(1000)
            self.status_message.emit("Game resumed")
    
    def log_message(self, message: str):
        """Add message to log."""
        self.log_widget.append(message)
    
    def on_game_started(self, event: GameEvent):
        """Handle game started event."""
        self.log_message("Game started!")
        self.update_display()
    
    def on_turn_started(self, event: GameEvent):
        """Handle turn started event."""
        # Reset card drawing state at start of new turn
        self.is_drawing_cards = False
        self.cards_drawn_this_turn = 0
        self.wild_card_drawn = False
        self.normal_card_drawn = False
        
        player_id = event.player_id
        game = self.game_controller.game
        player = game.players[player_id]
        self.log_message(f"Player {player_id + 1} ({player.color}) turn started")
        self.update_display()
    
    def on_turn_ended(self, event: GameEvent):
        """Handle turn ended event."""
        # Reset card drawing state at end of turn
        self.is_drawing_cards = False
        self.cards_drawn_this_turn = 0
        self.wild_card_drawn = False
        self.normal_card_drawn = False
        self.update_display()
    
    def on_route_claimed(self, event: GameEvent):
        """Handle route claimed event."""
        self.update_display()
    
    def on_game_ended(self, event: GameEvent):
        """Handle game ended event."""
        game = self.game_controller.game
        winner = game.winner
        self.log_message(f"Game ended! Winner: Player {winner.player_id + 1} ({winner.color})")
        self.status_message.emit(f"Game Over - {winner.color} wins!")
    
    def request_ticket_selection(self, player, min_keep: int, tickets):
        """Request ticket selection from GUI (called by HumanPlayer).
        
        Args:
            player: The human player requesting selection
            min_keep: Minimum tickets to keep
            tickets: List of tickets to choose from
            
        Returns:
            Tuple[List[int], List[int]]: (kept_indices, discarded_indices)
        """
        dialog = TicketSelectionDialog(tickets, min_keep, parent=self.parent_window)
        result = dialog.exec()
        
        # Get selection immediately and store
        kept_indices, discarded_indices = dialog.get_selection()
        
        # Delete dialog reference
        del dialog
        
        if result:
            # Validate indices
            kept_indices = [idx for idx in kept_indices if 0 <= idx < len(tickets)]
            discarded_indices = [idx for idx in discarded_indices if 0 <= idx < len(tickets)]
            
            # Ensure minimum
            if len(kept_indices) < min_keep:
                kept_indices = list(range(min_keep))
                discarded_indices = [i for i in range(len(tickets)) if i not in kept_indices]
            
            return kept_indices, discarded_indices
        else:
            # User cancelled - keep minimum required
            return list(range(min_keep)), list(range(min_keep, len(tickets)))
    
    def request_route(self, player) -> int:
        """Request route selection from GUI (called by HumanPlayer).
        
        This is called from player.claim_route() after the user has clicked on a route.
        The route ID should already be stored in pending_route_id from handle_route_click().
        
        Args:
            player: The human player requesting route selection
            
        Returns:
            int: Route link ID
        """
        # The route should already be stored when handle_route_click was called
        if self._pending_route_decision is not None:
            return self._pending_route_decision
        
        # Fallback: return pending_route_id if available
        if self.pending_route_id is not None:
            return self.pending_route_id
        
        # If not set, return 0 (will fail validation in claim_route)
        return 0
    
    def request_card_color(self, player) -> int:
        """Request card color selection from GUI (called by HumanPlayer for grey routes).
        
        Args:
            player: The human player requesting color selection
            
        Returns:
            int: Index of selected color in TRAIN_COLORS list
        """
        game = self.game_controller.game
        available_colors = game.config.TRAIN_COLORS
        
        dialog = CardColorDialog(available_colors, parent=self.parent_window)
        color_index = dialog.get_selected_color_index()
        
        return color_index
    
    def request_wild_cards(self, player) -> int:
        """Request wild cards count from GUI (called by HumanPlayer).
        
        Args:
            player: The human player requesting wild cards count
            
        Returns:
            int: Number of wild cards to use
        """
        # Get route information to determine max wild cards needed
        # The route should be set when this is called from claim_route()
        route_id = self._pending_route_decision if self._pending_route_decision is not None else self.pending_route_id
        
        if route_id is None:
            return 0
        
        game = self.game_controller.game
        try:
            route_data = game.board.get_route_data(route_id)
            if route_data is None:
                return 0
            
            city1, city2, route_info = route_data
            route_length = route_info['weight']
            max_wild_cards = min(player.hand.get('wild', 0), route_length)
            
            dialog = WildCardsDialog(max_wild_cards, route_length, parent=self.parent_window)
            wild_count = dialog.get_wild_cards_count()
            
            return wild_count
        except Exception as e:
            self.log_message(f"Error getting route data for wild cards: {e}")
            return 0

