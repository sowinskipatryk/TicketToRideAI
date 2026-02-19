class NetworkDecisions:
    # 120 decision ids
    ROUTE_DECISION_ID = 0  # 100 route decisions (id 0-99)
    COLOR_DECISION_ID = 100  # 8 color decisions (id 100-107)
    TRAIN_CARD_DECISION_ID = 108  # 6 train card decisions: 5 face up + 1 draw pile (id 108-113)
    TICKET_DECISION_ID = 114  # 1 ticket decision, all tickets chosen separately (id 114)
    ACTION_DECISION_ID = 115  # 4 action decisions - cards, tickets, routes, skip (id 115-118)
    LOCOMOTIVE_DECISION_ID = 119  # 1 locomotive decision, how many locomotive cards to use (id 119)
