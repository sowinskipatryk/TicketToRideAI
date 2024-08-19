from game_logic.game import Game

if __name__ == "__main__":
    # Provide a list of 2-5 players. Choices: Human, AI, Random
    players = ['Random', 'Random']

    # Choose version of the game. Choices: USA, Europe, Nordic
    version = 'USA'

    game = Game(players, version)
    game.play()
