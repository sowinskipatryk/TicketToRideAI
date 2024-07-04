import random

from game_config import GameConfig


class TrainCardManager:
    def __init__(self):
        self.draw_pile = self._create_draw_pile()
        self.discard_pile = []
        self.face_up_cards = []

        random.shuffle(self.draw_pile)
        self.fill_face_up()

    @staticmethod
    def _create_draw_pile():
        deck = [x for x in GameConfig.TRAIN_COLORS for _ in range(GameConfig.TRAIN_CARDS_NUM)]
        deck += ["wild" for _ in range(GameConfig.WILD_CARDS_NUM)]
        return deck

    def get_draw_pile(self):
        return self.draw_pile

    def get_face_up_cards(self):
        return self.face_up_cards

    def get_discard_pile(self):
        return self.discard_pile

    def pick_face_up_card(self, card):
        assert card in self.draw_pile
        self.face_up_cards.remove(card)
        self.fill_face_up()
        return card

    def pick_draw_pile_card(self):
        if len(self.draw_pile) == 0:
            self.fill_draw_pile()

        if len(self.draw_pile) != 0:
            return self.draw_pile.pop()

    def fill_face_up(self):
        while len(self.face_up_cards) < GameConfig.FACE_UP_CARDS_NUM:
            card = self.pick_draw_pile_card()
            if card is None:
                break
            else:
                self.face_up_cards.append(card)

            if self.face_up_cards.count('wild') >= GameConfig.MAX_WILD_CARDS:
                self.add_to_discard_pile(self.face_up_cards)
                self.face_up_cards = []

    def fill_draw_pile(self):
        assert len(self.draw_pile) == 0
        if len(self.discard_pile) != 0:
            self.draw_pile = self.discard_pile
            random.shuffle(self.draw_pile)
            self.discard_pile = []

    def add_to_discard_pile(self, cards):
        for card in cards:
            self.discard_pile.append(card)
