class BaseAdapter:
    def __init__(self):
        self.state_array = []

    def set_face_up_card(self, card_pos, color_id):
        pass

    def set_draw_pile_num(self, value):
        pass

    def set_discard_pile_num(self, value):
        pass

    def set_ticket_pile_num(self, value):
        pass

    def set_trains_num(self, player_id, value):
        pass

    def set_route_owner(self, player_id, route_id):
        pass

    def set_wild_cards_num(self, player_id, value):
        pass

    def set_color_cards_num(self, player_id, color, value):
        pass

    def set_ticket_owner(self, player_id, ticket_id):
        pass

    def set_ticket_completed(self, player_id, ticket_id):
        pass

    def get_state_array(self, player_id):
        pass
