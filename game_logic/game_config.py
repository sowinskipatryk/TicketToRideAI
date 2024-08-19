class BaseGameConfig:
    TICKETS_DEALT_NUM = 3
    TICKETS_TO_KEEP_NUM = 1
    INITIAL_TICKETS_DEALT_NUM = 3
    INITIAL_TICKETS_TO_KEEP_NUM = 2
    TRAIN_FIGURES_NUM = 45
    WILD_CARDS_NUM = 14
    TRAIN_CARDS_NUM = 12  # per color
    TRAIN_COLORS = ["red", "blue", "green", "yellow", "black", "white", "pink", "orange"]
    TRAIN_CARDS_DEALT_NUM = 2
    MAX_WILD_CARDS = 3
    FACE_UP_CARDS_NUM = 5
    STARTING_HAND_SIZE = 4
    MIN_TRAIN_FIGURES_NUM = 3  # triggers last round
    ROUTE_VALUES = {1: 1, 2: 2, 3: 4, 4: 7, 5: 10, 6: 15, 8: 21, 9: 27}  # {length: value}


class GameConfigUSA(BaseGameConfig):
    WILD_CARD_RESTRICTION = True
    LONGEST_ROUTE_BONUS = 10


class GameConfigEurope(BaseGameConfig):
    WILD_CARD_RESTRICTION = True
    LONGEST_ROUTE_BONUS = 10
    STATIONS_NUM = 15


class GameConfigNordic(BaseGameConfig):
    WILD_CARD_RESTRICTION = False
    GLOBETROTTER_BONUS = 10
