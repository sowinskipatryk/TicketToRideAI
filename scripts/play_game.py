from game import Game

if __name__ == "__main__":

    # Provide a list of 2-5 players. Choices: Human, NEAT, Random
    players = ['Random', 'Random']

    # Choose version of the game. Choices: USA, Europe
    version = 'USA'

    game = Game(players, version)
    game.play()
