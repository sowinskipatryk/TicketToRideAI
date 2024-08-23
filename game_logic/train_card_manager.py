import random
from typing import List, Optional, TYPE_CHECKING

from game_logic.game_logger import logger

if TYPE_CHECKING:
    from game_logic.game import Game


class TrainCardManager:
    WILD_CARD = 'wild'

    def __init__(self, game_instance: 'Game') -> None:
        self.game_instance = game_instance

        self._draw_pile = self._create_draw_pile()
        self.set_draw_pile_num_adapter()

        self._discard_pile = []
        self.set_discard_pile_num_adapter()

        self._face_up_cards = [None] * 5
        self.fill_face_up()

    def _create_draw_pile(self) -> List[str]:
        deck = [x for x in self.game_instance.config.TRAIN_COLORS for _ in range(self.game_instance.config.TRAIN_CARDS_NUM)]
        deck += [self.WILD_CARD for _ in range(self.game_instance.config.WILD_CARDS_NUM)]
        random.shuffle(deck)
        return deck

    def get_face_up_cards(self) -> List[Optional[str]]:
        return self._face_up_cards

    def get_discard_pile(self) -> List[str]:
        return self._discard_pile

    def get_face_up_cards_num(self) -> int:
        return sum(1 for card in self._face_up_cards if card is not None)

    def pick_face_up_card(self, card_id: int) -> Optional[str]:
        if card_id < self.get_face_up_cards_num():
            card = self._face_up_cards[card_id]
            logger.debug(f'pick_face_up: {card}')
            if card is None:
                return
            self._face_up_cards[card_id] = None
            self.set_face_up_card_adapter(card_id, None)

            self.fill_face_up()
            return card

    def pick_draw_pile_card(self) -> str:
        if len(self._draw_pile) == 0:
            self.fill_draw_pile()

        if len(self._draw_pile) != 0:
            card = self._draw_pile.pop()
            self.set_draw_pile_num_adapter()
            return card

    def fill_face_up(self) -> None:
        logger.debug('fill_face_up')
        logger.debug(f'before {self._face_up_cards}')
        self.get_state()
        tries = 0
        while self.get_face_up_cards_num() < self.game_instance.config.FACE_UP_CARDS_NUM:
            closest_none_id = self._face_up_cards.index(None)
            card = self.pick_draw_pile_card()
            if card is None:
                logger.debug(f'Draw pile is empty!')
                break
            else:
                # logger.debug(f'{card} picked from draw pile')
                self._face_up_cards[closest_none_id] = card
                self.set_face_up_card_adapter(closest_none_id, card)

            if self._face_up_cards.count(self.WILD_CARD) >= self.game_instance.config.MAX_WILD_CARDS and tries < 5:
                logger.debug('Too many wild cards on the table. Discarding face up cards...')
                cards = [card for card in self._face_up_cards if card is not None]
                self.add_to_discard_pile(cards)
                self._face_up_cards = [None] * 5
                tries += 1
        logger.debug(f'after {self._face_up_cards}')
        self.get_state()

    def fill_draw_pile(self) -> None:
        logger.debug('fill_draw_pile')
        self.get_state()
        if len(self._discard_pile) != 0:
            self.fill_from_discard_pile()
        else:
            logger.debug('Discard pile is empty!')
        self.get_state()

    def fill_from_discard_pile(self):
        self._draw_pile = self._discard_pile
        random.shuffle(self._draw_pile)

        self._discard_pile = []
        self.set_discard_pile_num_adapter()

    def add_to_discard_pile(self, cards: List[str]) -> None:
        if cards:
            logger.debug(f'add_to_discard: {cards}')
        for card in cards:
            self._discard_pile.append(card)
        self.set_discard_pile_num_adapter()

    def get_state(self) -> None:
        logger.debug(f'face_up: {len(self._face_up_cards)} draw: {len(self._draw_pile)} discard: {len(self._discard_pile)}')

    def set_draw_pile_num_adapter(self):
        self.game_instance.adapter.set_draw_pile_num(len(self._draw_pile))

    def set_discard_pile_num_adapter(self):
        self.game_instance.adapter.set_discard_pile_num(len(self._discard_pile))

    def set_face_up_card_adapter(self, card_id, card):
        if card == self.WILD_CARD:
            color_id = len(self.game_instance.config.TRAIN_COLORS)
        elif card is None:
            color_id = len(self.game_instance.config.TRAIN_COLORS) + 1
        else:
            color_id = self.game_instance.config.TRAIN_COLORS.index(card)
        self.game_instance.adapter.set_face_up_card(card_id, color_id)
