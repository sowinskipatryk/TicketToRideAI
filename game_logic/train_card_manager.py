import random
from typing import List

from game_logic.game_config import GameConfigAmerica
from game_logic.game_logger import logger


class TrainCardManager:
    def __init__(self):
        self._draw_pile = self._create_draw_pile()
        self._discard_pile = []
        self._face_up_cards = []

        random.shuffle(self._draw_pile)
        self.fill_face_up()

    @staticmethod
    def _create_draw_pile() -> List[str]:
        deck = [x for x in GameConfigAmerica.TRAIN_COLORS for _ in range(GameConfigAmerica.TRAIN_CARDS_NUM)]
        deck += ["wild" for _ in range(GameConfigAmerica.WILD_CARDS_NUM)]
        return deck

    def get_draw_pile(self) -> List[str]:
        return self._draw_pile

    def get_face_up_cards(self) -> List[str]:
        return self._face_up_cards

    def get_discard_pile(self) -> List[str]:
        return self._discard_pile

    def pick_face_up_card(self, card_id: int) -> str:
        if card_id < len(self._face_up_cards):
            card = self._face_up_cards.pop(card_id)
            logger.debug(f'pick_face_up: {card}')
            self.fill_face_up()
            return card

    def pick_draw_pile_card(self) -> str:
        if len(self._draw_pile) == 0:
            self.fill_draw_pile()

        if len(self._draw_pile) != 0:
            card = self._draw_pile.pop()
            return card

    def fill_face_up(self) -> None:
        logger.debug('fill_face_up')
        logger.debug(f'before {self._face_up_cards}')
        self.get_state()
        while len(self._face_up_cards) < GameConfigAmerica.FACE_UP_CARDS_NUM:
            card = self.pick_draw_pile_card()
            if card is None:
                logger.debug(f'Draw pile is empty!')
                break
            else:
                # logger.debug(f'{card} picked from draw pile')
                self._face_up_cards.append(card)

            if self._face_up_cards.count('wild') >= GameConfigAmerica.MAX_WILD_CARDS and len(self._discard_pile) > 0:
                logger.debug('Too many wild cards on the table. Discarding face up cards...')
                self.add_to_discard_pile(self._face_up_cards)
                self._face_up_cards = []
        logger.debug(f'after {self._face_up_cards}')
        self.get_state()

    def fill_draw_pile(self) -> None:
        logger.debug('fill_draw_pile')
        self.get_state()
        assert len(self._draw_pile) == 0
        if len(self._discard_pile) != 0:
            self._draw_pile = self._discard_pile
            random.shuffle(self._draw_pile)
            self._discard_pile = []
        else:
            logger.debug('Discard pile is empty!')
        self.get_state()

    def add_to_discard_pile(self, cards: List[str]) -> None:
        if cards:
            logger.debug(f'add_to_discard: {cards}')
        for card in cards:
            self._discard_pile.append(card)

    def get_state(self):
        logger.debug(f'face_up: {len(self._face_up_cards)} draw: {len(self._draw_pile)} discard: {len(self._discard_pile)}')
