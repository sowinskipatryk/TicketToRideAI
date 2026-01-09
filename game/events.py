"""Event system for game state changes."""
from dataclasses import dataclass
from typing import Any, Optional, Dict, List
from enum import Enum

from game.enums import ActionDecision, PlayerColor
from game.ticket_deck import Ticket


class EventType(Enum):
    """Types of game events."""
    GAME_STARTED = "game_started"
    GAME_ENDED = "game_ended"
    TURN_STARTED = "turn_started"
    TURN_ENDED = "turn_ended"
    ROUTE_CLAIMED = "route_claimed"
    CARDS_DRAWN = "cards_drawn"
    TICKETS_DRAWN = "tickets_drawn"
    TICKET_COMPLETED = "ticket_completed"
    SCORE_CHANGED = "score_changed"
    PLAYER_ACTION = "player_action"
    INVALID_MOVE = "invalid_move"
    STATE_CHANGED = "state_changed"


@dataclass
class GameEvent:
    """Represents a game event."""
    event_type: EventType
    player_id: Optional[int] = None
    data: Optional[Dict[str, Any]] = None
    timestamp: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'event_type': self.event_type.value,
            'player_id': self.player_id,
            'data': self.data,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GameEvent':
        """Create from dictionary."""
        return cls(
            event_type=EventType(data['event_type']),
            player_id=data.get('player_id'),
            data=data.get('data'),
            timestamp=data.get('timestamp', 0.0)
        )


class EventBus:
    """Event bus for observer pattern."""
    
    def __init__(self):
        self._subscribers: Dict[EventType, List[callable]] = {}
        self._event_history: List[GameEvent] = []
        self._enabled = True
    
    def subscribe(self, event_type: EventType, callback: callable):
        """Subscribe to an event type.
        
        Args:
            event_type: Type of event to subscribe to
            callback: Function to call when event occurs
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)
    
    def unsubscribe(self, event_type: EventType, callback: callable):
        """Unsubscribe from an event type."""
        if event_type in self._subscribers:
            try:
                self._subscribers[event_type].remove(callback)
            except ValueError:
                pass
    
    def emit(self, event: GameEvent):
        """Emit an event to all subscribers.
        
        Args:
            event: The event to emit
        """
        if not self._enabled:
            return
        
        # Store event in history
        self._event_history.append(event)
        
        # Notify subscribers
        if event.event_type in self._subscribers:
            for callback in self._subscribers[event.event_type]:
                try:
                    callback(event)
                except Exception as e:
                    # Log error but don't break event system
                    import traceback
                    print(f"Error in event callback: {e}")
                    traceback.print_exc()
    
    def get_event_history(self) -> List[GameEvent]:
        """Get all events in order."""
        return self._event_history.copy()
    
    def clear_history(self):
        """Clear event history."""
        self._event_history.clear()
    
    def enable(self):
        """Enable event system."""
        self._enabled = True
    
    def disable(self):
        """Disable event system."""
        self._enabled = False
    
    def save_history(self, filepath: str):
        """Save event history to file."""
        import json
        events_data = [e.to_dict() for e in self._event_history]
        with open(filepath, 'w') as f:
            json.dump(events_data, f, indent=2)
    
    def load_history(self, filepath: str):
        """Load event history from file."""
        import json
        with open(filepath, 'r') as f:
            events_data = json.load(f)
        self._event_history = [GameEvent.from_dict(e) for e in events_data]

