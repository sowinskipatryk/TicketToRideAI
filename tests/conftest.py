# Ensure the game package (and its player_factory chain) is fully initialized
# before any test imports alphazero.encoding. Without this, importing
# alphazero.encoding first triggers a circular import:
#   alphazero.encoding → game.ticket_deck → game/__init__.py → game.core
#   → player_factory → neat_player → alphazero.encoding (partial, not ready)
import game.core  # noqa: F401
