"""Serializable game state for cloning, replay, and debugging."""
import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from collections import defaultdict

from game.enums import GameState as GameStateEnum, PlayerColor
from game.ticket_deck import Ticket


@dataclass
class PlayerState:
    """Serializable player state."""
    player_id: int
    color: PlayerColor
    tickets: Dict[Ticket, bool] = field(default_factory=dict)
    hand: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    score: int = 0
    trains_remaining: int = 45
    longest_path: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'player_id': self.player_id,
            'color': self.color.value,
            'tickets': {f"{t.city_from}-{t.city_to}": completed for t, completed in self.tickets.items()},
            'hand': dict(self.hand),
            'score': self.score,
            'trains_remaining': self.trains_remaining,
            'longest_path': self.longest_path
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], ticket_map: Dict[str, Ticket]) -> 'PlayerState':
        """Create from dictionary."""
        state = cls(
            player_id=data['player_id'],
            color=PlayerColor[data['color'].upper()],
            score=data['score'],
            trains_remaining=data['trains_remaining'],
            longest_path=data['longest_path']
        )
        state.hand = defaultdict(int, data['hand'])
        state.tickets = {
            ticket_map[ticket_key]: completed 
            for ticket_key, completed in data['tickets'].items()
        }
        return state


@dataclass
class BoardState:
    """Serializable board state."""
    claimed_routes: Dict[int, Optional[str]] = field(default_factory=dict)  # link_id -> player_color
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'claimed_routes': {str(k): v.value if v else None for k, v in self.claimed_routes.items()}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BoardState':
        """Create from dictionary."""
        state = cls()
        state.claimed_routes = {
            int(k): PlayerColor[v.upper()] if v else None 
            for k, v in data['claimed_routes'].items()
        }
        return state


@dataclass
class CardDeckState:
    """Serializable card deck state."""
    draw_pile: List[str] = field(default_factory=list)
    discard_pile: List[str] = field(default_factory=list)
    face_up_cards: List[Optional[str]] = field(default_factory=lambda: [None] * 5)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'draw_pile': self.draw_pile.copy(),
            'discard_pile': self.discard_pile.copy(),
            'face_up_cards': self.face_up_cards.copy()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CardDeckState':
        """Create from dictionary."""
        return cls(
            draw_pile=data['draw_pile'].copy(),
            discard_pile=data['discard_pile'].copy(),
            face_up_cards=data['face_up_cards'].copy()
        )


@dataclass
class TicketDeckState:
    """Serializable ticket deck state."""
    tickets_left: int = 0
    ticket_order: List[str] = field(default_factory=list)  # Ticket keys in order
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'tickets_left': self.tickets_left,
            'ticket_order': self.ticket_order.copy()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TicketDeckState':
        """Create from dictionary."""
        return cls(
            tickets_left=data['tickets_left'],
            ticket_order=data['ticket_order'].copy()
        )


@dataclass
class GameStateSnapshot:
    """Complete serializable game state snapshot."""
    version: str
    game_state: GameStateEnum
    current_player_id: int
    last_player_id: Optional[int]
    winner_id: Optional[int]
    players: List[PlayerState]
    board: BoardState
    card_deck: CardDeckState
    ticket_deck: TicketDeckState
    stats: Dict[str, List[int]]
    move_number: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'version': self.version,
            'game_state': self.game_state.name if isinstance(self.game_state, GameStateEnum) else str(self.game_state),
            'current_player_id': self.current_player_id,
            'last_player_id': self.last_player_id,
            'winner_id': self.winner_id,
            'players': [p.to_dict() for p in self.players],
            'board': self.board.to_dict(),
            'card_deck': self.card_deck.to_dict(),
            'ticket_deck': self.ticket_deck.to_dict(),
            'stats': self.stats.copy(),
            'move_number': self.move_number
        }
    
    def clone(self) -> 'GameStateSnapshot':
        """Create a deep copy of the state."""
        return copy.deepcopy(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], ticket_map: Dict[str, Ticket]) -> 'GameStateSnapshot':
        """Create from dictionary."""
        return cls(
            version=data['version'],
            game_state=GameStateEnum[data['game_state']],
            current_player_id=data['current_player_id'],
            last_player_id=data.get('last_player_id'),
            winner_id=data.get('winner_id'),
            players=[PlayerState.from_dict(p, ticket_map) for p in data['players']],
            board=BoardState.from_dict(data['board']),
            card_deck=CardDeckState.from_dict(data['card_deck']),
            ticket_deck=TicketDeckState.from_dict(data['ticket_deck']),
            stats=data['stats'].copy(),
            move_number=data.get('move_number', 0)
        )

