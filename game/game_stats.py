class GameStats:
    def __init__(self, game_instance):
        self.game_instance = game_instance
        self.invalid_moves = [0] * self.game_instance.players_num
        self.completed_moves = [0] * self.game_instance.players_num

    def add_invalid_move(self, player_id):
        self.invalid_moves[player_id] += 1

    def add_completed_move(self, player_id):
        self.completed_moves[player_id] += 1

    def get_completed_moves_sum(self):
        return sum(self.completed_moves)

    def get_invalid_moves_sum(self):
        return sum(self.invalid_moves)

    def get_total_moves(self):
        return self.invalid_moves + self.completed_moves

    def get_total_moves_sum(self):
        return sum(self.get_total_moves())
