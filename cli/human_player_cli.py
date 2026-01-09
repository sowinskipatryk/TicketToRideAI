"""CLI interface for human player interactions."""
from typing import List
from game.ticket_deck import Ticket


class HumanPlayerCLI:
    """CLI interface methods for human player."""
    
    @staticmethod
    def decide_tickets(min_keep: int, tickets: List[Ticket], game) -> tuple:
        """Get ticket selection from human player via CLI.
        
        Args:
            min_keep: Minimum number of tickets that must be kept
            tickets: List of tickets to choose from
            game: Game instance for state instructions
            
        Returns:
            Tuple[List[int], List[int]]: (kept_ticket_indices, discarded_ticket_indices)
        """
        while True:
            print("Choose tickets you intend to keep (separated by commas):")
            for i, ticket in enumerate(tickets):
                print(f"[{i + 1}] Ticket {ticket.city_from} - {ticket.city_to} ({ticket.points})")
            HumanPlayerCLI.print_state_instructions()
            decision = input("Enter your choice: ").strip()
            if HumanPlayerCLI.check_state_instructions(decision, game):
                continue
            ticket_choices = [c.strip() for c in decision.split(',') if c.strip()]
            if len(ticket_choices) >= min_keep and all(choice.isdigit() for choice in ticket_choices):
                chosen_ticket_ids = [int(i) - 1 for i in ticket_choices]
                # Validate all indices are in range
                if all(0 <= idx < len(tickets) for idx in chosen_ticket_ids):
                    # Remove duplicates while preserving order
                    seen = set()
                    unique_ids = []
                    for idx in chosen_ticket_ids:
                        if idx not in seen:
                            seen.add(idx)
                            unique_ids.append(idx)
                    discarded_ticket_ids = [i for i in range(len(tickets)) if i not in unique_ids]
                    return unique_ids, discarded_ticket_ids
                else:
                    print(f"Invalid ticket numbers. Please enter numbers between 1 and {len(tickets)}")
            else:
                print(f"Invalid choice. Please enter at least {min_keep} ticket number(s) separated by commas (e.g. 1,2,3)")
    
    @staticmethod
    def decide_action(game) -> int:
        """Get action decision from human player via CLI."""
        while True:
            print("Choose an action:")
            print("[1] Claim a route")
            print("[2] Draw tickets")
            print("[3] Draw train cards")
            HumanPlayerCLI.print_state_instructions()
            choice = input("Enter your choice: ")
            if HumanPlayerCLI.check_state_instructions(choice, game):
                continue
            elif choice in ['1', '2', '3']:
                return int(choice) - 1
            else:
                print("Invalid choice.")
    
    @staticmethod
    def decide_wild_cards(game) -> int:
        """Get wild card count from human player via CLI."""
        while True:
            wild_cards = input(f"How many wild cards do you wish to use (0-{game.config.NUM_WILD_CARDS}): ").strip()
            if wild_cards.isdigit():
                count = int(wild_cards)
                if 0 <= count <= game.config.NUM_WILD_CARDS:
                    return count
                else:
                    print(f"Invalid choice. Please enter a number between 0 and {game.config.NUM_WILD_CARDS}")
            else:
                print("Invalid input. Please enter a number.")
    
    @staticmethod
    def decide_cards_color(hand: dict, game) -> int:
        """Get card color decision from human player via CLI."""
        while True:
            print(hand)
            train_color = input("Which train color do you wish to use to claim a route: ")
            if train_color in game.config.TRAIN_COLORS:
                return game.config.TRAIN_COLORS.index(train_color)
            else:
                print(f"Invalid choice. Please enter one of these train colors: {game.config.TRAIN_COLORS}")
    
    @staticmethod
    def decide_train_card(game) -> int:
        """Get train card selection from human player via CLI."""
        face_up_cards = game.train_card_manager.get_face_up_cards()
        available_cards = [i for i, card in enumerate(face_up_cards) if card is not None]
        while True:
            print("Choose an action:")
            for i, card in enumerate(face_up_cards):
                if card is not None:
                    print(f"[{i + 1}] Choose revealed card: {card}")
            print(f"[{len(face_up_cards) + 1}] Draw a card from the deck")
            HumanPlayerCLI.print_state_instructions()
            chosen_train_card = input("Enter the index of the train card you want to take: ").strip()
            if HumanPlayerCLI.check_state_instructions(chosen_train_card, game):
                continue
            elif chosen_train_card.isdigit():
                chosen_action_index = int(chosen_train_card)
                max_choice = len(face_up_cards) + 1
                if 1 <= chosen_action_index <= max_choice:
                    # If choosing a face-up card, verify it exists
                    if chosen_action_index <= len(face_up_cards):
                        if face_up_cards[chosen_action_index - 1] is not None:
                            return chosen_action_index - 1
                        else:
                            print(f"Card position {chosen_action_index} is empty. Please choose an available card.")
                    else:
                        # Draw from deck
                        return len(face_up_cards)  # This maps to DRAW_PILE decision
            print(f"Invalid choice! Please enter a number from 1 to {len(face_up_cards) + 1}")
    
    @staticmethod
    def decide_route(game) -> int:
        """Get route selection from human player via CLI."""
        while True:
            HumanPlayerCLI.print_state_instructions()
            num_routes = game.board.get_route_links_num()
            route_decision = input("Enter the route id: ").strip()
            if HumanPlayerCLI.check_state_instructions(route_decision, game):
                continue
            elif route_decision.isdigit():
                route_id = int(route_decision)
                if 0 <= route_id < num_routes:
                    return route_id
                else:
                    print(f"Invalid choice. Please enter route id from 0 to {num_routes - 1}")
            else:
                print(f"Invalid input. Please enter a number from 0 to {num_routes - 1}")
    
    @staticmethod
    def print_hand(player, hand: dict):
        """Print player's hand."""
        print(f'{player} hand')
        for k, v in hand.items():
            print(f'{k}: {v}')
    
    @staticmethod
    def print_tickets(player, tickets: dict):
        """Print player's tickets."""
        print(f'{player} tickets')
        for k, v in tickets.items():
            print(f"{k.city_from} -> {k.city_to} ({k.points}) : {'finished' if v else 'not finished'}")
    
    @staticmethod
    def graph_time_decision(function):
        """Get graph display time from user."""
        while True:
            graph_decision = input("How long should the map be shown on the screen (0-60 seconds): ")
            if graph_decision.isdigit() and 0 <= int(graph_decision) <= 60:
                function(int(graph_decision))
                break
            else:
                print(f"Invalid choice. Please enter number from 0 to 60")
    
    @staticmethod
    def print_state_instructions():
        """Print state instruction options."""
        print("or choose one of the state options:")
        print("[p] Show possession graph")
        print("[m] Show moves graph")
        print("[h] Show player's hand")
        print("[t] Show player's tickets")
    
    @staticmethod
    def check_state_instructions(choice: str, game) -> bool:
        """Check and handle state instruction choices."""
        if choice == 'p':
            HumanPlayerCLI.graph_time_decision(game.board.draw_possession_graph)
            return True
        elif choice == 'm':
            HumanPlayerCLI.graph_time_decision(game.board.draw_available_moves_graph)
            return True
        elif choice == 'h':
            # Would need player reference - simplified for now
            return True
        elif choice == 't':
            # Would need player reference - simplified for now
            return True
        return False

