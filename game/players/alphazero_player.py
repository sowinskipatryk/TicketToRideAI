"""AlphaZero player: uses NN-guided MCTS for decision-making."""
import networkx as nx
import torch
from typing import Tuple, List, TYPE_CHECKING

from game.enums import TrainCardDecision
from game.players.base_player import BasePlayer
from game.ticket_deck import Ticket

if TYPE_CHECKING:
    from alphazero.network import AlphaZeroNet
    from alphazero.az_mcts import AlphaZeroMCTS


class AlphaZeroPlayer(BasePlayer):
    """Player that uses AlphaZero MCTS (NN-guided, no rollouts).

    Runs AZ-MCTS search on decide_action() and caches the chosen Action.
    Subsequent decision methods read from the cached action.
    """

    DEFAULT_ITERATIONS = 200

    def __init__(self, color_index, game, adapter, model_path=None, iterations=None,
                 device=None):
        super().__init__(color_index, game, adapter)

        # Lazy imports to avoid circular dependency
        from alphazero.network import AlphaZeroNet
        from alphazero.az_mcts import AlphaZeroMCTS

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.network = AlphaZeroNet()
        if model_path:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                self.network.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.network.load_state_dict(checkpoint)
        self.network.to(device)
        self.network.eval()

        iters = iterations or self.DEFAULT_ITERATIONS
        self.az_mcts = AlphaZeroMCTS(
            self.network, iterations=iters, c_puct=1.5,
            device=device, dirichlet_epsilon=0.0,  # No noise during play
        )
        self._chosen_action = None
        self._card_draw_count = 0

    def decide_action(self) -> int:
        action, _ = self.az_mcts.search(self.game, self.player_id, temperature=0.0)
        self._chosen_action = action
        self._card_draw_count = 0
        return action.action_type

    def decide_route(self) -> int:
        if self._chosen_action and self._chosen_action.action_type == 0:
            return self._chosen_action.link_id
        return 0

    def decide_cards_color(self) -> int:
        if self._chosen_action and self._chosen_action.color:
            try:
                return self.game.config.TRAIN_COLORS.index(self._chosen_action.color)
            except ValueError:
                pass
        return max(range(len(self.game.config.TRAIN_COLORS)),
                   key=lambda i: self.hand.get(self.game.config.TRAIN_COLORS[i], 0))

    def decide_wild_cards(self) -> int:
        if self._chosen_action and self._chosen_action.action_type == 0:
            return self._chosen_action.wilds
        return 0

    def decide_train_card(self) -> int:
        if self._card_draw_count == 0 and self._chosen_action and self._chosen_action.action_type == 2:
            self._card_draw_count += 1
            choice = self._chosen_action.card_choice
            if 0 <= choice <= 5:
                return choice
        self._card_draw_count += 1
        return TrainCardDecision.DRAW_PILE.value

    def decide_tickets(self, min_keep: int, tickets: List[Ticket]) -> Tuple[List[int], List[int]]:
        scored = []
        for i, ticket in enumerate(tickets):
            try:
                dist = nx.shortest_path_length(
                    self.game.board.G, ticket.city_from, ticket.city_to, weight='weight'
                )
                score = ticket.points / max(dist, 1)
                if dist <= self.trains_remaining:
                    score += 1
                else:
                    score -= 2
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                score = -1
            scored.append((i, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        keep_count = max(min_keep, sum(1 for _, s in scored if s > 0))
        keep_count = min(keep_count, len(tickets))

        indices = [i for i, _ in scored]
        return indices[:keep_count], indices[keep_count:]
