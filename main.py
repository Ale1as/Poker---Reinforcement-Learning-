import tkinter as tk
from tkinter import ttk
import random
from collections import defaultdict, Counter
from enum import IntEnum
from itertools import combinations
import threading
import time

class Suit(IntEnum):
    HEARTS = 0
    DIAMONDS = 1
    CLUBS = 2
    SPADES = 3

class Rank(IntEnum):
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14

class Card:
    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit
    
    def __repr__(self):
        rank_str = {11: 'J', 12: 'Q', 13: 'K', 14: 'A'}.get(self.rank, str(self.rank))
        suit_str = ['♥', '♦', '♣', '♠'][self.suit]
        return f"{rank_str}{suit_str}"
    
    def __eq__(self, other):
        return self.rank == other.rank and self.suit == other.suit
    
    def __hash__(self):
        return hash((self.rank, self.suit))

class HandRank(IntEnum):
    HIGH_CARD = 0
    PAIR = 1
    TWO_PAIR = 2
    THREE_OF_KIND = 3
    STRAIGHT = 4
    FLUSH = 5
    FULL_HOUSE = 6
    FOUR_OF_KIND = 7
    STRAIGHT_FLUSH = 8

class PokerHand:
    @staticmethod
    def evaluate(cards):
        ranks = sorted([c.rank for c in cards], reverse=True)
        suits = [c.suit for c in cards]
        rank_counts = Counter(ranks)
        
        is_flush = len(set(suits)) == 1
        is_straight = len(set(ranks)) == 5 and (max(ranks) - min(ranks) == 4)
        
        if set(ranks) == {14, 2, 3, 4, 5}:
            is_straight = True
            ranks = [5, 4, 3, 2, 1]
        
        counts = sorted(rank_counts.values(), reverse=True)
        unique_ranks = sorted(rank_counts.keys(), key=lambda x: (rank_counts[x], x), reverse=True)
        
        if is_straight and is_flush:
            return (HandRank.STRAIGHT_FLUSH, ranks[:1])
        elif counts == [4, 1]:
            return (HandRank.FOUR_OF_KIND, unique_ranks[:2])
        elif counts == [3, 2]:
            return (HandRank.FULL_HOUSE, unique_ranks[:2])
        elif is_flush:
            return (HandRank.FLUSH, ranks)
        elif is_straight:
            return (HandRank.STRAIGHT, ranks[:1])
        elif counts == [3, 1, 1]:
            return (HandRank.THREE_OF_KIND, unique_ranks[:3])
        elif counts == [2, 2, 1]:
            return (HandRank.TWO_PAIR, unique_ranks[:3])
        elif counts == [2, 1, 1, 1]:
            return (HandRank.PAIR, unique_ranks[:4])
        else:
            return (HandRank.HIGH_CARD, ranks)

class Action(IntEnum):
    FOLD = 0
    CALL = 1
    RAISE = 2

class PokerGame:
    def __init__(self, num_players=3, starting_chips=1000, small_blind=10):
        self.num_players = num_players
        self.starting_chips = starting_chips
        self.small_blind = small_blind
        self.big_blind = small_blind * 2
        self.reset()
    
    def reset(self):
        self.deck = [Card(rank, suit) for rank in Rank for suit in Suit]
        random.shuffle(self.deck)
        
        self.players_chips = [self.starting_chips] * self.num_players
        self.players_bet = [0] * self.num_players
        self.players_folded = [False] * self.num_players
        self.players_hands = [[] for _ in range(self.num_players)]
        self.community_cards = []
        self.pot = 0
        self.current_bet = self.big_blind
        self.dealer_pos = 0
        
        for i in range(self.num_players):
            self.players_hands[i] = [self.deck.pop(), self.deck.pop()]
        
        self.players_bet[1] = self.small_blind
        self.players_bet[2] = self.big_blind
        self.pot = self.small_blind + self.big_blind
        
        return self.get_state(0)
    
    def get_state(self, player_id):
        hand = self.players_hands[player_id]
        hand_strength = self._estimate_hand_strength(hand, self.community_cards)
        position = player_id / self.num_players
        pot_odds = self.current_bet / (self.pot + 1)
        chips_ratio = self.players_chips[player_id] / self.starting_chips
        
        return (
            round(hand_strength, 1),
            round(position, 1),
            round(pot_odds, 1),
            round(chips_ratio, 1),
            len(self.community_cards)
        )
    
    def _estimate_hand_strength(self, hand, community):
        if len(community) == 0:
            ranks = sorted([c.rank for c in hand], reverse=True)
            suited = hand[0].suit == hand[1].suit
            
            if ranks[0] == ranks[1]:
                return 0.5 + (ranks[0] / 14) * 0.3
            
            strength = (ranks[0] + ranks[1]) / 28
            if suited:
                strength += 0.1
            return min(strength, 1.0)
        
        all_cards = hand + community
        if len(all_cards) < 5:
            return 0.5
        
        best_rank = max([PokerHand.evaluate(list(combo)) 
                        for combo in combinations(all_cards, 5)])
        
        return (best_rank[0] / 8) + 0.2
    
    def get_valid_actions(self, player_id):
        if self.players_folded[player_id]:
            return []
        
        actions = [Action.FOLD, Action.CALL]
        
        if self.players_chips[player_id] > self.current_bet:
            actions.append(Action.RAISE)
        
        return actions
    
    def step(self, player_id, action):
        if action == Action.FOLD:
            self.players_folded[player_id] = True
            return self.get_state(player_id), 0, self._is_hand_over()
        
        elif action == Action.CALL:
            call_amount = min(self.current_bet - self.players_bet[player_id],
                            self.players_chips[player_id])
            self.players_chips[player_id] -= call_amount
            self.players_bet[player_id] += call_amount
            self.pot += call_amount
        
        elif action == Action.RAISE:
            raise_amount = self.big_blind
            total_needed = self.current_bet - self.players_bet[player_id] + raise_amount
            total_needed = min(total_needed, self.players_chips[player_id])
            
            self.players_chips[player_id] -= total_needed
            self.players_bet[player_id] += total_needed
            self.pot += total_needed
            self.current_bet = self.players_bet[player_id]
        
        if self._is_hand_over():
            reward = self._determine_winner(player_id)
            return self.get_state(player_id), reward, True
        
        return self.get_state(player_id), 0, False
    
    def _is_hand_over(self):
        active_players = sum(1 for f in self.players_folded if not f)
        
        if active_players <= 1:
            return True
        
        all_matched = all(
            self.players_bet[i] == self.current_bet or self.players_folded[i]
            for i in range(self.num_players)
        )
        
        if all_matched and len(self.community_cards) >= 5:
            return True
        
        return False
    
    def _determine_winner(self, player_id):
        active_players = [i for i, f in enumerate(self.players_folded) if not f]
        
        if len(active_players) == 1:
            winner = active_players[0]
            if winner == player_id:
                return self.pot - self.players_bet[player_id]
            else:
                return -self.players_bet[player_id]
        
        while len(self.community_cards) < 5:
            self.community_cards.append(self.deck.pop())
        
        hands_eval = []
        for i in active_players:
            hand_value = PokerHand.evaluate(self.players_hands[i] + self.community_cards)
            hands_eval.append((hand_value, i))
        
        hands_eval.sort(reverse=True, key=lambda x: (x[0][0], x[0][1]))
        winner = hands_eval[0][1]
        
        if winner == player_id:
            return self.pot - self.players_bet[player_id]
        else:
            return -self.players_bet[player_id]
    
    def deal_flop(self):
        if len(self.community_cards) == 0:
            self.deck.pop()
            self.community_cards.extend([self.deck.pop() for _ in range(3)])
    
    def deal_turn(self):
        if len(self.community_cards) == 3:
            self.deck.pop()
            self.community_cards.append(self.deck.pop())
    
    def deal_river(self):
        if len(self.community_cards) == 4:
            self.deck.pop()
            self.community_cards.append(self.deck.pop())

class QLearningAgent:
    def __init__(self, learning_rate=0.1, discount=0.95, epsilon=0.1):
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.lr = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
    
    def get_action(self, state, valid_actions, training=True):
        if training and random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        q_values = {a: self.q_table[state][a] for a in valid_actions}
        max_q = max(q_values.values())
        best_actions = [a for a, q in q_values.items() if q == max_q]
        return random.choice(best_actions)
    
    def update(self, state, action, reward, next_state, valid_next_actions, done):
        current_q = self.q_table[state][action]
        
        if done:
            max_next_q = 0
        else:
            max_next_q = max([self.q_table[next_state][a] for a in valid_next_actions], default=0)
        
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q

class SimpleOpponent:
    def __init__(self, strategy='tight'):
        self.strategy = strategy
    
    def get_action(self, game, player_id):
        hand = game.players_hands[player_id]
        community = game.community_cards
        valid_actions = game.get_valid_actions(player_id)
        
        if not valid_actions:
            return None
        
        strength = game._estimate_hand_strength(hand, community)
        
        if self.strategy == 'tight':
            if strength < 0.4:
                return Action.FOLD
            elif strength < 0.6:
                return Action.CALL if Action.CALL in valid_actions else Action.FOLD
            else:
                return Action.RAISE if Action.RAISE in valid_actions else Action.CALL
        
        elif self.strategy == 'loose':
            if strength < 0.3:
                return Action.FOLD
            elif random.random() < 0.3:
                return Action.RAISE if Action.RAISE in valid_actions else Action.CALL
            else:
                return Action.CALL if Action.CALL in valid_actions else Action.FOLD
        
        return random.choice(valid_actions)

class PokerVisualizerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🎰 Poker Reinforcement Learning")
        self.root.geometry("1400x900")
        self.root.configure(bg='#1e3c72')
        
        self.game = PokerGame()
        self.agent = QLearningAgent(learning_rate=0.1, discount=0.95, epsilon=0.2)
        self.opponents = [SimpleOpponent('tight'), SimpleOpponent('loose')]
        
        self.episode = 0
        self.total_episodes = 1000
        self.wins = 0
        self.rewards_history = []
        self.is_training = False
        self.training_speed = 50  # milliseconds
        
        self.setup_ui()
        
    def setup_ui(self):
        # Title
        title = tk.Label(self.root, text="🎰 Poker Reinforcement Learning 🃏", 
                        font=("Arial", 24, "bold"), bg='#1e3c72', fg='white')
        title.pack(pady=10)
        
        # Main container
        main_frame = tk.Frame(self.root, bg='#1e3c72')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Left side - Poker table
        left_frame = tk.Frame(main_frame, bg='#1e3c72')
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Poker table canvas
        self.canvas = tk.Canvas(left_frame, width=700, height=600, bg='#35654d', highlightthickness=2, highlightbackground='#FFD700')
        self.canvas.pack(pady=10)
        
        # Action log
        log_frame = tk.LabelFrame(left_frame, text="Action Log", font=("Arial", 12, "bold"), 
                                 bg='#2a5298', fg='white')
        log_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.log_text = tk.Text(log_frame, height=15, bg='#1e3c72', fg='white', font=("Courier", 10))
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Right side - Stats and controls
        right_frame = tk.Frame(main_frame, bg='#1e3c72')
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(20, 0))
        
        # Stats frame
        stats_frame = tk.LabelFrame(right_frame, text="Training Statistics", 
                                   font=("Arial", 12, "bold"), bg='#2a5298', fg='white')
        stats_frame.pack(fill=tk.X, pady=10)
        
        self.episode_label = tk.Label(stats_frame, text="Episode: 0/1000", 
                                     font=("Arial", 14), bg='#2a5298', fg='white')
        self.episode_label.pack(pady=5)
        
        self.wins_label = tk.Label(stats_frame, text="Wins: 0", 
                                  font=("Arial", 14), bg='#2a5298', fg='#FFD700')
        self.wins_label.pack(pady=5)
        
        self.winrate_label = tk.Label(stats_frame, text="Win Rate: 0.0%", 
                                     font=("Arial", 14), bg='#2a5298', fg='#00FF00')
        self.winrate_label.pack(pady=5)
        
        self.qtable_label = tk.Label(stats_frame, text="Q-Table Size: 0", 
                                    font=("Arial", 14), bg='#2a5298', fg='white')
        self.qtable_label.pack(pady=5)
        
        self.reward_label = tk.Label(stats_frame, text="Avg Reward: $0", 
                                    font=("Arial", 14), bg='#2a5298', fg='white')
        self.reward_label.pack(pady=5)
        
        # Chart canvas
        chart_frame = tk.LabelFrame(right_frame, text="Training Progress", 
                                   font=("Arial", 12, "bold"), bg='#2a5298', fg='white')
        chart_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.chart_canvas = tk.Canvas(chart_frame, width=350, height=250, bg='#1e3c72')
        self.chart_canvas.pack(padx=5, pady=5)
        
        # Controls frame
        control_frame = tk.Frame(right_frame, bg='#1e3c72')
        control_frame.pack(fill=tk.X, pady=10)
        
        self.train_btn = tk.Button(control_frame, text="Start Training", 
                                   command=self.start_training, 
                                   font=("Arial", 12, "bold"), bg='#4CAF50', fg='white',
                                   width=15, height=2)
        self.train_btn.pack(pady=5)
        
        speed_frame = tk.Frame(control_frame, bg='#1e3c72')
        speed_frame.pack(pady=5)
        
        tk.Label(speed_frame, text="Speed:", font=("Arial", 10), bg='#1e3c72', fg='white').pack(side=tk.LEFT)
        self.speed_var = tk.IntVar(value=50)
        speed_slider = tk.Scale(speed_frame, from_=10, to=200, orient=tk.HORIZONTAL, 
                               variable=self.speed_var, bg='#2a5298', fg='white',
                               highlightthickness=0, length=150)
        speed_slider.pack(side=tk.LEFT, padx=5)
        
        tk.Button(control_frame, text="Reset", command=self.reset_all,
                 font=("Arial", 10), bg='#f44336', fg='white', width=15).pack(pady=5)
        
        self.draw_table()
        
    def draw_table(self):
        self.canvas.delete("all")
        
        # Draw ellipse table
        self.canvas.create_oval(50, 100, 650, 500, fill='#2d5a3d', outline='#FFD700', width=3)
        
        # Draw players
        self.draw_player(0, 250, 470, "🤖 RL Agent")  # moved up slightly so cards are visible
        self.draw_player(1, 50, 200, "🎯 Tight Player")
        self.draw_player(2, 500, 200, "🎲 Loose Player")
        
        # Draw community cards
        self.draw_community_cards()
        
        # Draw pot
        pot_text = f"POT: ${self.game.pot}"
        self.canvas.create_rectangle(280, 280, 420, 320, fill='black', outline='#FFD700', width=2)
        self.canvas.create_text(350, 300, text=pot_text, font=("Arial", 16, "bold"), fill='#FFD700')
        
    def draw_player(self, player_id, x, y, name):
        # Player box
        color = '#4169E1' if not self.game.players_folded[player_id] else '#808080'
        self.canvas.create_rectangle(x, y, x+180, y+120, fill=color, outline='#FFD700', width=2)
        
        # Name
        self.canvas.create_text(x+90, y+15, text=name, font=("Arial", 10, "bold"), fill='white')
        
        # Chips
        chips_text = f"${self.game.players_chips[player_id]}"
        self.canvas.create_text(x+90, y+35, text=chips_text, font=("Arial", 12, "bold"), fill='#FFD700')
        
        # Bet
        if self.game.players_bet[player_id] > 0:
            bet_text = f"Bet: ${self.game.players_bet[player_id]}"
            self.canvas.create_text(x+90, y+55, text=bet_text, font=("Arial", 10), fill='white')
        
        # Cards
        if self.game.players_folded[player_id]:
            self.canvas.create_text(x+90, y+85, text="FOLDED", font=("Arial", 12, "bold"), fill='red')
        else:
            for i, card in enumerate(self.game.players_hands[player_id]):
                self.draw_card(card, x+20+i*70, y+65, hidden=(player_id != 0))
    
    def draw_card(self, card, x, y, hidden=False):
        # Card rectangle
        if hidden:
            self.canvas.create_rectangle(x, y, x+50, y+70, fill='gray', outline='black', width=2)
            self.canvas.create_text(x+25, y+35, text="?", font=("Arial", 20, "bold"), fill='white')
        else:
            self.canvas.create_rectangle(x, y, x+50, y+70, fill='white', outline='black', width=2)
            
            rank_str = {11: 'J', 12: 'Q', 13: 'K', 14: 'A'}.get(card.rank, str(card.rank))
            suit_str = ['♥', '♦', '♣', '♠'][card.suit]
            
            color = 'red' if card.suit in [Suit.HEARTS, Suit.DIAMONDS] else 'black'
            
            self.canvas.create_text(x+25, y+20, text=rank_str, font=("Arial", 14, "bold"), fill=color)
            self.canvas.create_text(x+25, y+50, text=suit_str, font=("Arial", 18), fill=color)
    
    def draw_community_cards(self):
        if not self.game.community_cards:
            return
        
        start_x = 250
        for i, card in enumerate(self.game.community_cards):
            self.draw_card(card, start_x + i*60, 240, hidden=False)
    
    def add_log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        if len(self.log_text.get("1.0", tk.END).split("\n")) > 100:
            self.log_text.delete("1.0", "2.0")
    
    def update_stats(self):
        self.episode_label.config(text=f"Episode: {self.episode}/{self.total_episodes}")
        self.wins_label.config(text=f"Wins: {self.wins}")
        
        win_rate = (self.wins / max(self.episode, 1)) * 100
        self.winrate_label.config(text=f"Win Rate: {win_rate:.1f}%")
        
        self.qtable_label.config(text=f"Q-Table Size: {len(self.agent.q_table)}")
        
        if self.rewards_history:
            avg_reward = sum(self.rewards_history[-10:]) / min(len(self.rewards_history), 10)
            self.reward_label.config(text=f"Avg Reward: ${avg_reward:.0f}")
    
    def draw_chart(self):
        self.chart_canvas.delete("all")
        
        if len(self.rewards_history) < 2:
            return
        
        # Draw axes
        self.chart_canvas.create_line(30, 220, 330, 220, fill='white', width=2)
        self.chart_canvas.create_line(30, 20, 30, 220, fill='white', width=2)
        
        # Plot rewards
        data = self.rewards_history[-50:]
        if not data:
            return
        
        max_val = max(max(data), 1)
        min_val = min(min(data), -1)
        range_val = max_val - min_val if max_val != min_val else 1
        
        points = []
        for i, reward in enumerate(data):
            x = 30 + (i / max(len(data) - 1, 1)) * 300
            y = 220 - ((reward - min_val) / range_val) * 200
            points.append((x, y))
        
        for i in range(len(points) - 1):
            self.chart_canvas.create_line(points[i][0], points[i][1], 
                                         points[i+1][0], points[i+1][1], 
                                         fill='#00FF00', width=2)
        
        # Labels
        self.chart_canvas.create_text(180, 10, text="Average Reward", 
                                     font=("Arial", 10, "bold"), fill='white')
    
    def train_step(self):
        if self.episode >= self.total_episodes:
            self.is_training = False
            self.train_btn.config(text="Start Training", state=tk.NORMAL)
            self.add_log("✅ Training Complete!")
            return
        
        state = self.game.reset()
        episode_reward = 0
        round_count = 0
        
        action_names = ['FOLD', 'CALL', 'RAISE']
        
        while round_count < 20:
            # Agent's turn
            valid_actions = self.game.get_valid_actions(0)
            if valid_actions:
                action = self.agent.get_action(state, valid_actions, training=True)
                self.add_log(f"🤖 Agent: {action_names[action]}")
                
                next_state, reward, done = self.game.step(0, action)
                
                if not done:
                    next_valid_actions = self.game.get_valid_actions(0)
                    self.agent.update(state, action, reward, next_state, next_valid_actions, done)
                    state = next_state
                else:
                    self.agent.update(state, action, reward, next_state, [], done)
                    episode_reward += reward
                    if reward > 0:
                        self.wins += 1
                        self.add_log(f"🎉 Agent wins ${reward:.0f}!")
                    break
            
            # Opponents
            for i, opponent in enumerate(self.opponents, start=1):
                if not self.game.players_folded[i]:
                    opp_action = opponent.get_action(self.game, i)
                    if opp_action is not None:
                        self.add_log(f"Player {i}: {action_names[opp_action]}")
                        _, _, done = self.game.step(i, opp_action)
                        if done:
                            break
            
            # Deal cards
            if len(self.game.community_cards) == 0:
                self.game.deal_flop()
                self.add_log("Dealing FLOP")
            elif len(self.game.community_cards) == 3:
                self.game.deal_turn()
                self.add_log("Dealing TURN")
            elif len(self.game.community_cards) == 4:
                self.game.deal_river()
                self.add_log("Dealing RIVER")
            
            round_count += 1
        
        self.episode += 1
        self.rewards_history.append(episode_reward)
        
        self.draw_table()
        self.update_stats()
        
        if self.episode % 10 == 0:
            self.draw_chart()
        
        if self.is_training:
            self.root.after(self.speed_var.get(), self.train_step)
    
    def start_training(self):
        if not self.is_training:
            self.is_training = True
            self.train_btn.config(text="Training...", state=tk.DISABLED)
            self.add_log("🚀 Training Started!")
            self.train_step()
    
    def reset_all(self):
        self.is_training = False
        self.episode = 0
        self.wins = 0
        self.rewards_history = []
        self.game.reset()
        self.agent = QLearningAgent(learning_rate=0.1, discount=0.95, epsilon=0.2)
        self.train_btn.config(text="Start Training", state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.add_log("System reset. Ready to train!")
        self.draw_table()
        self.update_stats()
        self.draw_chart()

if __name__ == "__main__":
    root = tk.Tk()
    app = PokerVisualizerGUI(root)
    root.mainloop()
