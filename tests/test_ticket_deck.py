"""Unit tests for ticket deck functionality."""
import pytest
from game.ticket_deck import TicketDeck, Ticket


class TestTicketDeck:
    """Test cases for TicketDeck class."""
    
    def test_ticket_creation(self):
        """Test ticket dataclass creation."""
        ticket = Ticket(city_from="New York", city_to="Los Angeles", points=21)
        assert ticket.city_from == "New York"
        assert ticket.city_to == "Los Angeles"
        assert ticket.points == 21
    
    def test_ticket_immutability(self):
        """Test that tickets are immutable (frozen dataclass)."""
        ticket = Ticket(city_from="A", city_to="B", points=5)
        with pytest.raises(Exception):  # Frozen dataclass raises exception on assignment
            ticket.points = 10


if __name__ == '__main__':
    pytest.main([__file__])

