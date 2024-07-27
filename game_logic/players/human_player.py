from game_logic.players.base_player import BasePlayer


class HumanPlayer(BasePlayer):
    def decide_action(self):
        raise NotImplementedError

    def decide_train_card(self):
        raise NotImplementedError

    def choose_tickets(self):
        raise NotImplementedError

    def decide_route(self):
        raise NotImplementedError

    def decide_cards_color(self):
        raise NotImplementedError

    def decide_ticket(self):
        raise NotImplementedError

    def decide_wild_cards(self):
        raise NotImplementedError
