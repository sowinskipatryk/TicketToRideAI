from enum import Enum, auto


class GameState(Enum):
    INIT = auto()
    RUNNING = auto()
    LAST_ROUND = auto()
    FINISHED = auto()


class GameVersion(Enum):
    USA = 'USA'
    EUROPE = 'Europe'
    NORDIC = 'Nordic'


class PlayerType(Enum):
    HUMAN = 'Human'
    NEAT = 'NEAT'
    RANDOM = 'Random'
    GREEDY = 'Greedy'
    TICKET_FOCUSED = 'TicketFocused'
    CARD_HOARDER = 'CardHoarder'
    BLOCKER = 'Blocker'
    MCTS = 'MCTS'


class PlayerColor(Enum):
    BLUE = 'blue'
    RED = 'red'
    GREEN = 'green'
    YELLOW = 'yellow'
    BLACK = 'black'

    @classmethod
    def from_index(cls, idx: int) -> "PlayerColor":
        return list(cls)[idx]


class ActionDecision(Enum):
    CLAIM_ROUTE = 0
    DRAW_TICKETS = 1
    DRAW_CARDS = 2
    SKIP = 3


class TrainCardDecision(Enum):
    FACE_UP1 = 0
    FACE_UP2 = 1
    FACE_UP3 = 2
    FACE_UP4 = 3
    FACE_UP5 = 4
    DRAW_PILE = 5
