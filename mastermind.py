import os
import random
import logging
import csv
import json
import datetime

# Configuration Section
PLAY_TYPE = 1  # 0 for player control, 1 for AI control
NUM_GAMES = 10  # Set the number of games to simulate
STRATEGIES = [0, 1, 2]  # An array of numbers representing the strategies to simulate this run
LOG_FILE = './logs/mastermind_game_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.log'
CSV_RESULTS_FILE = './results/mastermind_ai_results_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv'
JSON_RESULTS_FILE = './results/mastermind_ai_results_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.json'

# Configure logging
logging.basicConfig(filename=LOG_FILE, level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def define_strategy(strategy):
    """Translate the strategy into a user-friendly readable name."""
    if strategy == 0:
        return "Random Guess Strategy"
    elif strategy == 1:
        return "Simple Heuristic Strategy"
    elif strategy == 2:
        return "Knuth's Algorithm"
    else:
        logging.error(f"Strategy Definition Not Found for Strategy #{strategy}")
        return "Unknown Strategy"
    
class Mastermind:
    COLORS = ['RED', 'GREEN', 'BLUE', 'YELLOW', 'ORANGE', 'BLACK']

    def __init__(self):
        self.secret_code = self.generate_secret_code()
        self.attempts = 0
        self.max_attempts = 10
        self.guesses = []
        self.feedbacks = []

    def generate_secret_code(self):
        return [random.choice(self.COLORS) for _ in range(4)]

    def get_feedback(self, guess):
        black_pegs = sum(1 for i in range(4) if guess[i] == self.secret_code[i])
        white_pegs = sum(min(guess.count(color), self.secret_code.count(color)) for color in set(self.COLORS)) - black_pegs
        return black_pegs, white_pegs

    def play_turn(self, strategy):
        if strategy == 0:
            guess = [random.choice(self.COLORS) for _ in range(4)]
        elif strategy == 1:
            guess = self.simple_heuristic()
        elif strategy == 2:
            guess = self.knuth_algorithm()
        else:
            guess = [random.choice(self.COLORS) for _ in range(4)]

        black_pegs, white_pegs = self.get_feedback(guess)
        self.guesses.append(guess)
        self.feedbacks.append((black_pegs, white_pegs))
        self.attempts += 1
        return black_pegs == 4

    def simple_heuristic(self):
        # Implement a simple heuristic strategy
        return [random.choice(self.COLORS) for _ in range(4)]

    def knuth_algorithm(self):
        # Implement Knuth's algorithm for solving Mastermind
        return [random.choice(self.COLORS) for _ in range(4)]

    def play_game(self, strategy):
        while self.attempts < self.max_attempts:
            if self.play_turn(strategy):
                return True
        return False

    def get_results(self):
        return {
            'attempts': self.attempts,
            'guesses': self.guesses,
            'feedbacks': self.feedbacks,
            'secret_code': self.secret_code
        }
        
def simulate_games(num_games, strategies):
    results = []
    for strategy in strategies:
        for i in range(num_games):
            logging.debug(f"Starting game {i+1} with strategy {define_strategy(strategy)}")
            game = Mastermind()
            success = game.play_game(strategy)
            result = game.get_results()
            result.update({
                'game_number': i + 1,
                'strategy': strategy,
                'success': success
            })
            results.append(result)
            logging.debug(f"Game {i+1} ended with {result['attempts']} attempts")
    return results

def save_results_to_csv(results):
    if not os.path.exists(CSV_RESULTS_FILE):
        with open(CSV_RESULTS_FILE, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Strategy', 'Game Number', 'Attempts', 'Success', 'Guesses', 'Feedbacks', 'Secret Code'])
    with open(CSV_RESULTS_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        for result in results:
            writer.writerow([result['strategy'], result['game_number'], result['attempts'], result['success'], result['guesses'], result['feedbacks'], result['secret_code']])
    logging.debug(f'Results appended to {CSV_RESULTS_FILE}')

def save_results_to_json(results):
    if os.path.exists(JSON_RESULTS_FILE):
        with open(JSON_RESULTS_FILE, 'r') as file:
            existing_data = json.load(file)
    else:
        existing_data = []
    existing_data.extend(results)
    with open(JSON_RESULTS_FILE, 'w') as file:
        json.dump(existing_data, file, indent=4)
    logging.debug(f'Results saved to {JSON_RESULTS_FILE}')

if __name__ == '__main__':
    logging.debug(f"#### Beginning {len(STRATEGIES)} Strategy Testing ####")
    print("#### Beginning", len(STRATEGIES), "Strategy Tests ####")
    for strategy in STRATEGIES:
        logging.debug(f"## ## Simulation beginning for {NUM_GAMES} games using strategy #{strategy} ## ##")
        results = simulate_games(NUM_GAMES, [strategy])
        save_results_to_csv(results)
        save_results_to_json(results)
        avg_attempts = sum(result['attempts'] for result in results) / NUM_GAMES
        success_rate = sum(result['success'] for result in results) / NUM_GAMES * 100
        logging.debug(f"## ## Simulation ended for {NUM_GAMES} games using strategy #{define_strategy(strategy)} ## ##")
        print(f"Strategy Deployed: {define_strategy(strategy)}, Games Simulated: {NUM_GAMES}, Average Attempts: {round(avg_attempts, 2)}, Success Rate: {round(success_rate, 2)}%")
    print("#### All Simulations Have Ended ####")
    logging.debug(f"#### #### All Simulations Have Ended #### ####")
