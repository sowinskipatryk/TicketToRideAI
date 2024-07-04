from enum import Enum


class GameState(Enum):
    INIT = 1
    PLAYING = 2
    LAST_ROUND = 3
    FINISHED = 4
