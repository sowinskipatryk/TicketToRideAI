"""Game controller with state management and event system."""
import time
import random
from collections import defaultdict
from typing import List, Dict, Optional, Any
from neat.nn.feed_forward import FeedForwardNetwork

from game.game_logger import logger
from game.enums import GameState
from game.events import EventType, EventBus, GameEvent
from game.game_state import GameStateSnapshot, PlayerState, BoardState, CardDeckState, TicketDeckState
from game.core import Game
from game.ticket_deck import Ticket


class GameController:
    """Controller that wraps Game with state management and events.
    
    Provides:
    - State snapshots for cloning/replay
    - Event system for UI updates
    - State restoration for debugging/testing
    """
    
    def __init__(self, player_types: List[str], version: str, 
                 networks: List[FeedForwardNetwork] = None,
                 event_bus: Optional[EventBus] = None,
                 ui_interface=None):
        """Initialize game controller.
        
        Args:
            player_types: List of player type strings
            version: Game version
            networks: Optional NEAT networks
            event_bus: Optional event bus (creates new one if None)
            ui_interface: Optional UI interface for human players (GUI or CLI)
        """
        self.ui_interface = ui_interface
        self.game = Game(player_types, version, networks, ui_interface=ui_interface)
        self.event_bus = event_bus or EventBus()
        self._snapshots: List[GameStateSnapshot] = []
        self._current_move = 0
        
        # Emit initial event
        self.event_bus.emit(GameEvent(
            EventType.GAME_STARTED,
            data={'version': version, 'players': player_types}
        ))
    
    def get_state_snapshot(self) -> GameStateSnapshot:
        """Create a snapshot of current game state.
        
        Returns:
            GameStateSnapshot of current state
        """
        # Build player states
        player_states = []
        for player in self.game.players:
            ps = PlayerState(
                player_id=player.player_id,
                color=player.color,
                tickets=player.tickets.copy(),
                hand=defaultdict(int, player.hand),
                score=player.score,
                trains_remaining=player.trains_remaining,
                longest_path=player.longest_path
            )
            player_states.append(ps)
        
        # Build board state
        board_state = BoardState()
        for u, v, data in self.game.board.G.edges(data=True):
            link_id = data['link_id']
            claimed_by = data.get('claimed_by')
            board_state.claimed_routes[link_id] = claimed_by
        
        # Build card deck state
        card_state = CardDeckState(
            draw_pile=self.game.train_card_manager._draw_pile.copy(),
            discard_pile=self.game.train_card_manager._discard_pile.copy(),
            face_up_cards=self.game.train_card_manager._face_up_cards.copy()
        )
        
        # Build ticket deck state
        ticket_state = TicketDeckState(
            tickets_left=self.game.ticket_deck.tickets_left,
            ticket_order=[f"{t.city_from}-{t.city_to}" for t in self.game.ticket_deck._ticket_deck]
        )
        
        return GameStateSnapshot(
            version=self.game.version,
            game_state=self.game.game_state,
            current_player_id=self.game.current_player_id,
            last_player_id=self.game.last_player.player_id if self.game.last_player else None,
            winner_id=self.game.winner.player_id if self.game.winner else None,
            players=player_states,
            board=board_state,
            card_deck=card_state,
            ticket_deck=ticket_state,
            stats=self.game.stats.copy(),
            move_number=self._current_move
        )
    
    def save_snapshot(self):
        """Save current state as a snapshot."""
        snapshot = self.get_state_snapshot()
        self._snapshots.append(snapshot)
        return len(self._snapshots) - 1  # Return snapshot index
    
    def restore_snapshot(self, snapshot: GameStateSnapshot):
        """Restore game state from snapshot.
        
        Note: This is a simplified restoration. Full restoration would require
        recreating the Game object, which is complex. This is mainly for
        read-only state inspection.
        
        Args:
            snapshot: State snapshot to restore
        """
        # This is a read-only operation for now
        # Full restoration would require refactoring Game.__init__ to accept state
        logger.warning("Full state restoration not yet implemented. Use for inspection only.")
        return snapshot
    
    def complete_turn(self, move_completed: bool = True, snapshot_idx: Optional[int] = None) -> None:
        """Complete the current turn and advance to next player.
        
        This method handles turn completion logic that is shared between
        CLI, GUI, and headless modes. It checks for last round conditions,
        moves to the next player, and handles game state transitions.
        
        Args:
            move_completed: Whether the current turn was completed successfully
            snapshot_idx: Optional snapshot index to include in TURN_ENDED event
        """
        current_player = self.game.get_current_player()
        
        if move_completed:
            self._current_move += 1
            event_data = {'move_number': self._current_move}
            if snapshot_idx is not None:
                event_data['snapshot_idx'] = snapshot_idx
            self.event_bus.emit(GameEvent(
                EventType.TURN_ENDED,
                player_id=current_player.player_id,
                data=event_data,
                timestamp=time.time()
            ))
            
            # Check for last round condition
            if self.game.last_round_condition(current_player):
                self.game.last_player = current_player
                if self.game.game_state == GameState.RUNNING:
                    self.game.game_state = GameState.LAST_ROUND
                    self.event_bus.emit(GameEvent(
                        EventType.STATE_CHANGED,
                        data={'new_state': GameState.LAST_ROUND.name}
                    ))
            
            # Move to next player (unless game is finished)
            if self.game.game_state != GameState.FINISHED:
                # Check if we should end the game (last round and back to last player)
                if self.game.game_state == GameState.LAST_ROUND:
                    if current_player is self.game.last_player:
                        self.game.game_state = GameState.FINISHED
                        self.event_bus.emit(GameEvent(
                            EventType.STATE_CHANGED,
                            data={'new_state': GameState.FINISHED.name}
                        ))
                    else:
                        self.game.move_to_next_player()
                else:
                    self.game.move_to_next_player()
        else:
            self.event_bus.emit(GameEvent(
                EventType.INVALID_MOVE,
                player_id=current_player.player_id,
                data={'move_number': self._current_move},
                timestamp=time.time()
            ))
    
    def play_turn(self) -> bool:
        """Play a single turn with event tracking.
        
        Returns:
            True if turn completed, False if invalid move
        """
        current_player = self.game.get_current_player()
        
        # Emit turn started event
        self.event_bus.emit(GameEvent(
            EventType.TURN_STARTED,
            player_id=current_player.player_id,
            data={'move_number': self._current_move},
            timestamp=time.time()
        ))
        
        # Save snapshot before move
        snapshot_idx = self.save_snapshot()
        
        # Play the turn
        move_completed = current_player.play_turn()
        
        # Complete the turn (handles events, state transitions, and player advancement)
        self.complete_turn(move_completed, snapshot_idx=snapshot_idx)
        
        return move_completed
    
    def play(self, max_moves: int = 0) -> Dict:
        """Play the game with full event tracking.
        
        Args:
            max_moves: Maximum moves (0 = no limit)
            
        Returns:
            Game statistics dictionary
        """
        # Initialize game
        self.game.ticket_deck.set_ticket_pile_num_adapter()
        self.game.game_state = GameState.RUNNING
        
        for player in self.game.players:
            player.draw_initial_train_cards(self.game.config.NUM_TRAIN_CARDS_DEALT_INIT)
            player.draw_tickets(
                num_tickets=self.game.config.NUM_TICKETS_DEALT_INIT,
                min_keep=self.game.config.MIN_TICKETS_KEPT_INIT
            )
        
        # Main game loop
        while self.game.game_state != GameState.LAST_ROUND:
            if max_moves and sum(self.game.stats['total_moves']) >= max_moves:
                break
            
            # play_turn() already handles turn completion and player advancement
            if not self.play_turn():
                continue
        
        # Last round
        while self.game.game_state != GameState.FINISHED:
            if max_moves and sum(self.game.stats['total_moves']) >= max_moves:
                break
            
            # play_turn() already handles turn completion and player advancement
            if not self.play_turn():
                continue
        
        # Score and determine winner
        self.game.score_player_tickets()
        self.game.score_longest_path()
        self.game.determine_winner()
        
        # Emit game ended event
        self.event_bus.emit(GameEvent(
            EventType.GAME_ENDED,
            player_id=self.game.winner.player_id if self.game.winner else None,
            data={'stats': self.game.stats.copy()},
            timestamp=time.time()
        ))
        
        return self.game.stats
    
    def get_snapshot(self, index: int) -> Optional[GameStateSnapshot]:
        """Get snapshot by index."""
        if 0 <= index < len(self._snapshots):
            return self._snapshots[index]
        return None
    
    def get_all_snapshots(self) -> List[GameStateSnapshot]:
        """Get all saved snapshots."""
        return self._snapshots.copy()
    
    def clone_state(self) -> GameStateSnapshot:
        """Create a clone of current state for AI use."""
        return self.get_state_snapshot().clone()

