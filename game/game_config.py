class BaseConfig:
    NUM_TICKETS_DEALT = 3
    MIN_TICKETS_KEPT = 1
    NUM_TICKETS_DEALT_INIT = 3
    MIN_TICKETS_KEPT_INIT = 2
    NUM_TRAIN_FIGURES = 45
    MIN_TRAIN_FIGURES = 3  # triggers last round
    NUM_WILD_CARDS = 14
    MAX_WILD_CARDS = 3
    NUM_TRAIN_CARDS_DEALT = 2
    NUM_TRAIN_CARDS_DEALT_INIT = 4
    NUM_TRAIN_CARDS_PER_COLOR = 12
    TRAIN_COLORS = ["red", "blue", "green", "yellow", "black", "white", "pink", "orange"]
    NUM_FACE_UP_CARDS = 5
    ROUTE_VALUES = {1: 1, 2: 2, 3: 4, 4: 7, 5: 10, 6: 15, 8: 21, 9: 27}  # {length: value}


class USAConfig(BaseConfig):
    WILD_CARD_RESTRICTION = True
    LONGEST_ROUTE_BONUS = 10


class EuropeConfig(BaseConfig):
    WILD_CARD_RESTRICTION = True
    LONGEST_ROUTE_BONUS = 10
    NUM_STATIONS = 15


class NordicConfig(BaseConfig):
    WILD_CARD_RESTRICTION = False
    GLOBETROTTER_BONUS = 10
