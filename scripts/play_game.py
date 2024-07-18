from game_logic.game import Game

if __name__ == "__main__":

    # Provide a list of players using words 'Human', 'AI', or 'Random'
    players = ['Random', 'Random', 'Random', 'Random', 'Random']
    
    game = Game(players)
    game.play()
