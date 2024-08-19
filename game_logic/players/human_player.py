from game_logic.players.base_player import BasePlayer


class HumanPlayer(BasePlayer):
    def choose_tickets(self, min_keep, tickets):
        while True:
            print("Choose tickets you intend to keep (separated by commas):")
            for i, ticket in enumerate(tickets):
                print(f"[{i + 1}] Ticket {ticket[0]} - {ticket[1]} ({ticket[2]})")
            self.print_state_instructions()
            decision = input("Enter your choice: ")
            if self.check_state_instructions(decision):
                continue
            ticket_choices = decision.split(',')
            if len(ticket_choices) >= min_keep and all(choice.isdigit() for choice in ticket_choices):
                chosen_ticket_ids = [int(i) - 1 for i in ticket_choices]
                discarded_ticket_ids = [i for i in range(len(tickets)) if i not in ticket_choices]
                return chosen_ticket_ids, discarded_ticket_ids
            else:
                print(f"Invalid choice. Please enter at least {min_keep} ticket number(s) separated by commas (e.g. 1,2,3)")

    def decide_action(self):
        while True:
            print("Choose an action:")
            print("[1] Claim a route")
            print("[2] Draw tickets")
            print("[3] Draw train cards")
            self.print_state_instructions()
            choice = input("Enter your choice: ")
            if self.check_state_instructions(choice):
                continue
            elif choice in ['1', '2', '3']:
                return int(choice) - 1
            else:
                print("Invalid choice.")

    def decide_wild_cards(self) -> int:
        while True:
            wild_cards = input(f"How many wild cards do you wish to use: ")
            if wild_cards.isdigit() and 0 <= int(wild_cards) <= self.game_instance.config.WILD_CARDS_NUM:
                return int(wild_cards)
            else:
                print("Invalid choice. Please enter the number of wild cards that you wish to use to claim the route.")

    def decide_cards_color(self) -> int:
        while True:
            print(self.hand)
            train_color = input("Which train color do you wish to use to claim a route: ")
            if train_color in self.game_instance.config.TRAIN_COLORS:
                return self.game_instance.config.TRAIN_COLORS.index(train_color)
            else:
                print(f"Invalid choice. Please enter one of these train colors: {self.game_instance.config.TRAIN_COLORS}")

    def decide_train_card(self):
        face_up_cards = self.game_instance.train_card_manager.get_face_up_cards()
        while True:
            print("Choose an action:")
            for i, card in enumerate(face_up_cards):
                print(f"[{i + 1}] Choose revealed card: {card}")
            print("[6] Draw a card from the deck")
            self.print_state_instructions()
            chosen_train_card = input("Enter the index of the train card you want to take: ")
            if self.check_state_instructions(chosen_train_card):
                continue
            elif chosen_train_card.isdigit():
                chosen_action_index = int(chosen_train_card)
                allowed_decisions = list(range(1, len(face_up_cards) + 1)) + [6]
                if chosen_action_index in allowed_decisions:
                    return chosen_action_index - 1
            print("Invalid choice! Please enter a number from 1 to 6 based on the action you intend to take")

    def decide_route(self):
        while True:
            self.print_state_instructions()
            route_decision = input("Enter the route id: ")
            if self.check_state_instructions(route_decision):
                continue
            elif route_decision.isdigit() and int(route_decision) in range(77):
                return int(route_decision)
            else:
                print(f"Invalid choice. Please enter route id from 0 to {self.game_instance.board.get_links_num()}")

    def print_hand(self):
        print(f'{self} hand')
        for k, v in self.hand.items():
            print(f'{k}: {v}')

    def print_tickets(self):
        print(f'{self} tickets')
        for k, v in self.tickets.items():
            print(f"{k[0]} -> {k[1]} ({k[2]}) : {'finished' if v else 'not finished'}")

    @staticmethod
    def graph_time_decision(function):
        while True:
            graph_decision = input("How long should the map be shown on the screen (0-60 seconds): ")
            if graph_decision.isdigit() and 0 <= int(graph_decision) <= 60:
                function(int(graph_decision))
                break
            else:
                print(f"Invalid choice. Please enter number from 0 to 60")

    @staticmethod
    def print_state_instructions():
        print("or choose one of the state options:")
        print("[p] Show possession graph")
        print("[m] Show moves graph")
        print("[h] Show player's hand")
        print("[t] Show player's tickets")

    def check_state_instructions(self, choice):
        if choice == 'p':
            self.graph_time_decision(self.game_instance.board.draw_possession_graph)
            return True
        elif choice == 'm':
            self.graph_time_decision(self.game_instance.board.draw_available_moves_graph)
            return True
        elif choice == 'h':
            self.print_hand()
            return True
        elif choice == 't':
            self.print_tickets()
            return True
