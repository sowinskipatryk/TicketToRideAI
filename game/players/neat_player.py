import math

from neat.nn import FeedForwardNetwork

from game.game_logger import logger
from game.players.base_player import BasePlayer
from neuroevolution.adapters.blank_adapter import BlankAdapter
from neuroevolution.decisions import NetworkDecisions


class NEATPlayer(BasePlayer):
    """NEAT-based AI player that uses neural networks to make decisions."""
    
    # Threshold for binary decisions (e.g., ticket keeping)
    TICKET_DECISION_THRESHOLD = 0.5
    
    def __init__(self, color_index: int, game: 'Game', adapter: BlankAdapter, network: FeedForwardNetwork = None):
        super().__init__(color_index, game, adapter)
        self.network = network

    def decide_route(self):
        decision_array = self.get_decision_array()
        logger.debug(f'route values: {decision_array[NetworkDecisions.ROUTE_DECISION_ID:NetworkDecisions.COLOR_DECISION_ID]}')
        route_values = decision_array[NetworkDecisions.ROUTE_DECISION_ID:NetworkDecisions.COLOR_DECISION_ID]
        return self.get_max_value_index(route_values)

    def decide_cards_color(self):
        decision_array = self.get_decision_array()
        color_values = decision_array[NetworkDecisions.COLOR_DECISION_ID:NetworkDecisions.TRAIN_CARD_DECISION_ID]
        return self.get_max_value_index(color_values)

    def decide_train_card(self):
        decision_array = self.get_decision_array()
        card_values = decision_array[NetworkDecisions.TRAIN_CARD_DECISION_ID:NetworkDecisions.TICKET_DECISION_ID]
        logger.debug(f'card decision values: {decision_array[NetworkDecisions.TRAIN_CARD_DECISION_ID:NetworkDecisions.TICKET_DECISION_ID]}')
        return self.get_max_value_index(card_values)

    def decide_ticket(self):
        decision_array = self.get_decision_array()
        ticket_value = decision_array[NetworkDecisions.TICKET_DECISION_ID]
        logger.debug(f'ticket decision value: {decision_array[NetworkDecisions.TICKET_DECISION_ID]}')
        return self.is_active(ticket_value, self.TICKET_DECISION_THRESHOLD)

    def decide_action(self):
        decision_array = self.get_decision_array()
        action_values = decision_array[NetworkDecisions.ACTION_DECISION_ID:NetworkDecisions.LOCOMOTIVE_DECISION_ID]
        logger.debug(f'action decision values: {decision_array[NetworkDecisions.ACTION_DECISION_ID:NetworkDecisions.LOCOMOTIVE_DECISION_ID]}')
        return self.get_max_value_index(action_values)

    def decide_wild_cards(self):
        decision_array = self.get_decision_array()
        locomotive_value = decision_array[NetworkDecisions.LOCOMOTIVE_DECISION_ID]
        return math.floor(locomotive_value * self.hand.get('wild', 0))

    def decide_tickets(self, min_keep, tickets):
        ticket_decision_values = [(i, self.decide_ticket()) for i in range(len(tickets))]
        chosen_tickets_num = sum(self.is_active(value) for _, value in ticket_decision_values)
        kept_tickets_num = max(min_keep, chosen_tickets_num)
        ticket_decision_values.sort(key=lambda x: x[1], reverse=True)
        ticket_decision_indices = [i for i, v in ticket_decision_values]
        return ticket_decision_indices[:kept_tickets_num], ticket_decision_indices[kept_tickets_num:]

    def get_input_array(self):
        return self.adapter.get_state_array(self.player_id)

    def get_decision_array(self):
        return self.network.activate(self.get_input_array())

    @staticmethod
    def get_max_value_index(values_list):
        max_value = max(values_list)
        return values_list.index(max_value)

    @staticmethod
    def is_active(value: float, threshold: float = TICKET_DECISION_THRESHOLD) -> int:
        """Determine if a value is active based on threshold.
        
        Args:
            value: The value to check
            threshold: The threshold value (defaults to class constant)
            
        Returns:
            1 if value >= threshold, 0 otherwise
        """
        return 1 if value >= threshold else 0
