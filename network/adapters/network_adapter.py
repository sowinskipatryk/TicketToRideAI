from game.game_logger import logger
from network.adapters.base_adapter import BaseAdapter


class NetworkAdapter(BaseAdapter):
    """
    Common values:
    - one-hot face-up cards x (8 colors + wild card) (5x9=45)
    - draw pile num (1)
    - discard pile num (1)
    - ticket pile num (1)
    - train figures num x players num (4)
    - one-hot routes x players num (100x4=400)

    Player values:
    - wild cards num (1)
    - train color cards num x colors (8)
    - one-hot ticket owned x tickets num (30)
    - one-hot ticket completed x tickets num (30)
    """

    VALUE_POSITIVE = 1.
    VALUE_NEUTRAL = 0.

    def __init__(self, game_instance):
        super().__init__()
        self.game_instance = game_instance
        self.train_colors_num = len(self.game_instance.config.TRAIN_COLORS)

        self.common_values = ((([self.VALUE_NEUTRAL] * self.game_instance.config.NUM_FACE_UP_CARDS * (self.train_colors_num + 1)
                                + [self.VALUE_POSITIVE] * 3)
                              + [self.VALUE_POSITIVE] * self.game_instance.players_num)
                              + [self.VALUE_NEUTRAL] * self.game_instance.board.get_route_links_num() * self.game_instance.players_num)

        self.player_values = [([self.VALUE_NEUTRAL]
                               + [self.VALUE_NEUTRAL] * self.train_colors_num
                               + [self.VALUE_NEUTRAL] * (self.game_instance.ticket_deck.get_tickets_num() * 2)
                               ) for _ in range(self.game_instance.players_num)]

        self.face_up_cards_pos = 0
        self.draw_pile_num_pos = self.game_instance.config.NUM_FACE_UP_CARDS * (self.train_colors_num + 1)
        self.discard_pile_num_pos = self.draw_pile_num_pos + 1
        self.ticket_pile_num_pos = self.discard_pile_num_pos + 1
        self.trains_num_pos = self.ticket_pile_num_pos + 1
        self.route_owner_pos = self.trains_num_pos + self.game_instance.players_num

        self.wild_cards_num_pos = 0
        self.color_cards_num_pos = self.wild_cards_num_pos + 1
        self.ticket_owner_pos = self.color_cards_num_pos + self.train_colors_num
        self.ticket_completed_pos = self.ticket_owner_pos + self.game_instance.ticket_deck.get_tickets_num()

    def normalize_trains_num(self, trains_num):
        return trains_num / self.game_instance.config.NUM_TRAIN_FIGURES

    def normalize_train_pile_num(self, cards_num):
        return cards_num / ((self.game_instance.config.NUM_TRAIN_CARDS_PER_COLOR * self.train_colors_num)
                            + self.game_instance.config.NUM_WILD_CARDS)

    def normalize_ticket_pile_num(self, cards_num):
        return cards_num / self.game_instance.ticket_deck.get_tickets_num()

    def normalize_color_cards_num(self, cards_num):
        return cards_num / self.game_instance.config.NUM_TRAIN_CARDS_PER_COLOR

    def normalize_wild_cards_num(self, cards_num):
        return cards_num / self.game_instance.config.NUM_WILD_CARDS

    def normalize_player(self, player_id):
        return player_id / (self.game_instance.players_num - 1)

    def set_face_up_card(self, card_pos, color_id):
        self.reset_face_up_card(card_pos)
        self.common_values[self.face_up_cards_pos + (card_pos * (self.train_colors_num + 1)) + color_id] = self.VALUE_POSITIVE

    def reset_face_up_card(self, card_pos):
        for i in range(self.train_colors_num + 1):
            self.common_values[self.face_up_cards_pos + (card_pos * (self.train_colors_num + 1)) + i] = self.VALUE_NEUTRAL

    def set_draw_pile_num(self, value):
        self.common_values[self.draw_pile_num_pos] = self.normalize_train_pile_num(value)

    def set_discard_pile_num(self, value):
        self.common_values[self.discard_pile_num_pos] = self.normalize_train_pile_num(value)

    def set_ticket_pile_num(self, value):
        self.common_values[self.ticket_pile_num_pos] = self.normalize_ticket_pile_num(value)

    def set_trains_num(self, player_id, value):
        self.common_values[self.trains_num_pos + player_id] = self.normalize_trains_num(value)

    def set_route_owner(self, player_id, route_id):
        self.common_values[self.route_owner_pos + (route_id * self.game_instance.players_num) + player_id] = self.VALUE_POSITIVE

    def set_wild_cards_num(self, player_id, value):
        self.player_values[player_id][self.wild_cards_num_pos] = self.normalize_wild_cards_num(value)

    def set_color_cards_num(self, player_id, color, value):
        color_id = self.game_instance.config.TRAIN_COLORS.index(color)
        self.player_values[player_id][self.color_cards_num_pos + color_id] = self.normalize_color_cards_num(value)

    def set_ticket_owner(self, player_id, ticket):
        ticket_id = self.game_instance.ticket_deck.get_ticket_id(ticket)
        self.player_values[player_id][self.ticket_owner_pos + ticket_id] = self.VALUE_POSITIVE

    def set_ticket_completed(self, player_id, ticket):
        ticket_id = self.game_instance.ticket_deck.get_ticket_id(ticket)
        self.player_values[player_id][self.ticket_completed_pos + ticket_id] = self.VALUE_POSITIVE

    def get_state_array(self, player_id):
        logger.debug(f"player {player_id} {self.common_values + self.player_values[player_id]}")
        return self.common_values + self.player_values[player_id]

    """
    def print_human_readable_information(self, player_id):
        logger.warning('Common values')
        logger.warning(f"table: {self.game_instance.config.TRAIN_COLORS + ['None', 'wild']}")
        logger.warning(f"face up: {self.game_instance.train_card_manager.get_face_up_cards()}")
        for i in range(self.game_instance.config.FACE_UP_CARDS_NUM):
            logger.warning(f'one hot card {i}: {self.common_values[(self.train_colors_num + 1) * i : (self.train_colors_num + 1) * (i + 1)]}')
        for i in range(self.game_instance.board.get_route_links_num()):
            logger.warning(f"route owner: {self.game_instance.board.get_link_owner(i)}")
            logger.warning(f"one hot route owned {i}: {self.common_values[self.route_owner_pos + (i * self.game_instance.players_num):self.route_owner_pos + ((i+1) * self.game_instance.players_num)]}")
        logger.warning(f"draw pile num: {len(self.game_instance.train_card_manager._draw_pile)}")
        logger.warning(f"draw pile norm: {self.common_values[self.draw_pile_num_pos]}")
        logger.warning(f"discard pile num: {len(self.game_instance.train_card_manager._discard_pile)}")
        logger.warning(f"discard pile norm: {self.common_values[self.discard_pile_num_pos]}")
        logger.warning(f"ticket deck num: {self.game_instance.ticket_deck.tickets_left}")
        logger.warning(f"ticket deck norm: {self.common_values[self.ticket_pile_num_pos]}")
        logger.warning(f"train figures num: {[player.trains_remaining for player in self.game_instance.players]}")
        logger.warning(f"train figures norm: {self.common_values[self.trains_num_pos : self.trains_num_pos + self.game_instance.players_num]}")
        logger.warning(f"hand: {self.game_instance.players[player_id].hand}")
        logger.warning(f"wild cards norm: {[self.player_values[player_id][self.wild_cards_num_pos]]}")
        logger.warning(f"table: {self.game_instance.config.TRAIN_COLORS}")
        logger.warning(f"color cards norm: {self.player_values[player_id][self.color_cards_num_pos:self.color_cards_num_pos + self.train_colors_num]}")
        logger.warning(f"tickets owned: {[self.game_instance.ticket_deck.get_ticket_id(ticket) for ticket in self.game_instance.players[player_id].tickets]}")
        logger.warning(f"one hot tickets owned: {self.player_values[player_id][self.ticket_owner_pos:self.ticket_owner_pos + self.game_instance.ticket_deck.tickets_num]}")
    """
