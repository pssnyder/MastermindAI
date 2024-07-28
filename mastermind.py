import os
import random
import logging
import csv
import json
import datetime
import itertools

# Configuration Section
PLAY_TYPE = 1  # 0 for player control, 1 for AI control
NUM_GAMES = 1000  # Set the number of games to simulate
STRATEGIES = [0, 1, 2, 3, 4, 5, 6]  # An array of numbers representing the strategies to simulate this run
LOG_FILE = './logs/mastermind_game_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.log'
CSV_RESULTS_FILE = './results/mastermind_ai_results_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv'
JSON_RESULTS_FILE = './results/mastermind_ai_results_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.json'

# Configure logging
logging.basicConfig(filename=LOG_FILE, level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def define_strategy(strategy):
    """Translate the strategy into a user-friendly readable name."""
    if strategy == 0:
        return "Monkey's w/ Typewriters"
    elif strategy == 1:
        return "Feedback-Based"
    elif strategy == 2:
        return "Comprehensive Feedback"
    elif strategy == 3:
        return "Knuth's Algorithm"
    elif strategy == 4:
        return "Average Case"
    elif strategy == 5:
        return "Minimax"
    elif strategy == 6:
        return "Genetic Algorithm"
    elif strategy == 7:
        return "Optimal Logical"
    else:
        logging.error(f"Strategy Definition Not Found for Strategy #{strategy}")
        return "Unknown Strategy"

def simple_heuristic(self):
    # Implement a simple heuristic strategy
    return [random.choice(self.COLORS) for _ in range(4)]

def random_guess_strategy():
    """Strategy 0: Randomly guesses a code from the available colors."""
    return [random.choice(Mastermind.COLORS) for _ in range(4)]

def feedback_based_strategy(previous_guesses, previous_feedbacks):
    """
    Strategy 1: Guesses based on feedback from the last move.

    Args:
        previous_guesses (list): List of previous guesses.
        previous_feedbacks (list): List of feedbacks for previous guesses.

    Returns:
        list: A new guess based on the feedback from the last move.
    """
    if not previous_guesses:
        return random_guess_strategy()

    last_guess = previous_guesses[-1]
    last_feedback = previous_feedbacks[-1]
    new_guess = last_guess[:]

    # Lock positions for black pegs
    for i in range(4):
        if last_feedback[0] > 0:
            new_guess[i] = last_guess[i]
            last_feedback = (last_feedback[0] - 1, last_feedback[1])
        else:
            break

    # Rotate remaining positions
    remaining_positions = [i for i in range(4) if new_guess[i] != last_guess[i]]
    for i in remaining_positions:
        new_guess[i] = last_guess[(i + 1) % 4]

    # Swap positions for white pegs
    if len(remaining_positions) >= 2:
        for _ in range(last_feedback[1]):
            pos1, pos2 = random.sample(remaining_positions, 2)
            new_guess[pos1], new_guess[pos2] = new_guess[pos2], new_guess[pos1]

    return new_guess

def comprehensive_feedback_strategy(previous_guesses, previous_feedbacks):
    """
    Strategy 2: Guesses based on feedback from all previous moves.

    Args:
        previous_guesses (list): List of previous guesses.
        previous_feedbacks (list): List of feedbacks for previous guesses.

    Returns:
        list: A new guess based on comprehensive feedback from all previous moves.
    """
    if not previous_guesses:
        return random_guess_strategy()

    possible_positions = [set(range(4)) for _ in range(4)]
    for guess, feedback in zip(previous_guesses, previous_feedbacks):
        # Lock positions for black pegs
        for i in range(4):
            if feedback[0] > 0:
                possible_positions[i] = {i}
                feedback = (feedback[0] - 1, feedback[1])
            else:
                break

        # Remove impossible positions for white pegs
        for i in range(4):
            if feedback[1] > 0:
                for j in range(4):
                    if i != j and guess[i] == guess[j]:
                        possible_positions[i].discard(j)
                        feedback = (feedback[0], feedback[1] - 1)
            else:
                break

    new_guess = [None] * 4
    for i in range(4):
        if len(possible_positions[i]) == 1:
            new_guess[i] = previous_guesses[-1][list(possible_positions[i])[0]]
        else:
            new_guess[i] = random.choice(Mastermind.COLORS)

    return new_guess

def knuth_algorithm_strategy(previous_guesses, previous_feedbacks):
    """
    Implements Knuth's algorithm for solving Mastermind.

    Args:
        previous_guesses (list): List of previous guesses.
        previous_feedbacks (list): List of feedbacks for previous guesses.

    Returns:
        list: The next guess based on Knuth's algorithm.
    """
    COLORS = ['RED', 'GREEN', 'BLUE', 'YELLOW', 'ORANGE', 'BLACK']
    all_codes = list(itertools.product(COLORS, repeat=4))

    def get_feedback(secret_code, guess):
        black_pegs = sum(1 for i in range(4) if guess[i] == secret_code[i])
        white_pegs = sum(min(guess.count(color), secret_code.count(color)) for color in set(guess)) - black_pegs
        return black_pegs, white_pegs

    if not previous_guesses:
        # Generate initial guess (e.g., first guess can be ['RED', 'RED', 'GREEN', 'GREEN'])
        return ['RED', 'RED', 'GREEN', 'GREEN']

    # Filter possible codes based on feedback received from previous guesses
    possible_codes = all_codes[:]
    for guess, feedback in zip(previous_guesses, previous_feedbacks):
        possible_codes = [code for code in possible_codes if get_feedback(code, guess) == feedback]

    # Minimax technique to choose the next guess
    def minimize_worst_case(codes):
        min_max_count = float('inf')
        best_guess = None
        for guess in all_codes:
            feedback_counts = {}
            for code in codes:
                feedback = get_feedback(code, guess)
                if feedback not in feedback_counts:
                    feedback_counts[feedback] = 0
                feedback_counts[feedback] += 1
            max_count = max(feedback_counts.values())
            if max_count < min_max_count:
                min_max_count = max_count
                best_guess = guess
        return best_guess

    next_guess = minimize_worst_case(possible_codes)
    return next_guess

def average_case_strategy(previous_guesses, previous_feedbacks):
    """
    Implements the average case strategy for solving Mastermind.

    Args:
        previous_guesses (list): List of previous guesses.
        previous_feedbacks (list): List of feedbacks for previous guesses.

    Returns:
        list: The next guess based on the average case strategy.
    """
    COLORS = ['RED', 'GREEN', 'BLUE', 'YELLOW', 'ORANGE', 'BLACK']
    all_codes = list(itertools.product(COLORS, repeat=4))

    def get_feedback(secret_code, guess):
        black_pegs = sum(1 for i in range(4) if guess[i] == secret_code[i])
        white_pegs = sum(min(guess.count(color), secret_code.count(color)) for color in set(guess)) - black_pegs
        return black_pegs, white_pegs

    if not previous_guesses:
        # Generate initial random guess
        return random_guess_strategy()

    # Filter possible codes based on feedback received from the last guess
    possible_codes = all_codes[:]
    for guess, feedback in zip(previous_guesses, previous_feedbacks):
        possible_codes = [code for code in possible_codes if get_feedback(code, guess) == feedback]

    # Calculate the average number of remaining possibilities for each potential guess
    def calculate_average_score(guess):
        feedback_counts = {}
        for code in possible_codes:
            feedback = get_feedback(code, guess)
            if feedback not in feedback_counts:
                feedback_counts[feedback] = 0
            feedback_counts[feedback] += 1
        total_remaining = sum(count ** 2 for count in feedback_counts.values())
        return total_remaining / len(possible_codes)

    best_guess = min(all_codes, key=calculate_average_score)
    return best_guess

def minimax_strategy(previous_guesses, previous_feedbacks):
    """
    Implements the minimax strategy for solving Mastermind.

    Args:
        previous_guesses (list): List of previous guesses.
        previous_feedbacks (list): List of feedbacks for previous guesses.

    Returns:
        list: The next guess based on the minimax strategy.
    """
    COLORS = ['RED', 'GREEN', 'BLUE', 'YELLOW', 'ORANGE', 'BLACK']
    all_codes = list(itertools.product(COLORS, repeat=4))

    def get_feedback(secret_code, guess):
        black_pegs = sum(1 for i in range(4) if guess[i] == secret_code[i])
        white_pegs = sum(min(guess.count(color), secret_code.count(color)) for color in set(guess)) - black_pegs
        return black_pegs, white_pegs

    if not previous_guesses:
        # Generate initial random guess
        return random_guess_strategy()

    # Filter possible codes based on feedback received from the last guess
    possible_codes = all_codes[:]
    for guess, feedback in zip(previous_guesses, previous_feedbacks):
        possible_codes = [code for code in possible_codes if get_feedback(code, guess) == feedback]

    # Minimax technique to choose the next guess
    def minimize_worst_case(codes):
        min_max_count = float('inf')
        best_guess = None
        for guess in all_codes:
            max_count = 0
            feedback_counts = {}
            for code in codes:
                feedback = get_feedback(code, guess)
                if feedback not in feedback_counts:
                    feedback_counts[feedback] = 0
                feedback_counts[feedback] += 1
            max_count = max(feedback_counts.values())
            if max_count < min_max_count:
                min_max_count = max_count
                best_guess = guess
        return best_guess

    next_guess = minimize_worst_case(possible_codes)
    return next_guess

def genetic_algorithm_strategy(previous_guesses, previous_feedbacks, population_size=100, max_generations=100):
    """
    Strategy 6: Genetic algorithm strategy.
    
    Args:
        previous_guesses (list): List of previous guesses.
        previous_feedbacks (list): List of feedbacks for previous guesses.
        population_size (int): Number of guesses in the population.
        max_generations (int): Maximum number of generations to evolve.
    
    Returns:
        list: A new guess based on the genetic algorithm strategy.
    """
    if not previous_guesses:
        # If there are no previous guesses, generate a random initial guess
        return [random.choice(Mastermind.COLORS) for _ in range(4)]

    def generate_initial_population(size):
        """
        Generate an initial population of random guesses.
        
        Args:
            size (int): The size of the population.
        
        Returns:
            list: A list of random guesses.
        """
        return [[random.choice(Mastermind.COLORS) for _ in range(4)] for _ in range(size)]

    def fitness(guess):
        """
        Evaluate the fitness of a guess based on feedback from previous guesses.
        
        Args:
            guess (list): The guess to evaluate.
        
        Returns:
            int: The fitness score of the guess.
        """
        score = 0
        for prev_guess, (black_pegs, white_pegs) in zip(previous_guesses, previous_feedbacks):
            feedback = get_feedback(prev_guess, guess)
            score += abs(feedback[0] - black_pegs) + abs(feedback[1] - white_pegs)
        return score

    def get_feedback(secret_code, guess):
        """
        Get the feedback (black and white pegs) for a given guess.
        
        Args:
            secret_code (list): The secret code.
            guess (list): The guess to evaluate.
        
        Returns:
            tuple: The number of black and white pegs.
        """
        black_pegs = sum(1 for i in range(4) if guess[i] == secret_code[i])
        white_pegs = sum(min(guess.count(color), secret_code.count(color)) for color in set(Mastermind.COLORS)) - black_pegs
        return black_pegs, white_pegs

    def selection(population):
        """
        Select the best guesses from the population based on fitness.
        
        Args:
            population (list): The current population of guesses.
        
        Returns:
            list: The selected population.
        """
        population.sort(key=fitness)
        return population[:population_size // 2]

    def crossover(parent1, parent2):
        """
        Combine parts of two guesses to form a new guess.
        
        Args:
            parent1 (list): The first parent guess.
            parent2 (list): The second parent guess.
        
        Returns:
            list: The new guess formed by combining parts of the parents.
        """
        crossover_point = random.randint(1, 3)
        return parent1[:crossover_point] + parent2[crossover_point:]

    def mutate(guess):
        """
        Introduce random changes to a guess to maintain diversity.
        
        Args:
            guess (list): The guess to mutate.
        
        Returns:
            list: The mutated guess.
        """
        if random.random() < 0.1:  # Mutation probability
            index = random.randint(0, 3)
            guess[index] = random.choice(Mastermind.COLORS)
        return guess

    # Generate the initial population
    population = generate_initial_population(population_size)
    
    # Evolve the population over multiple generations
    for generation in range(max_generations):
        selected_population = selection(population)
        new_population = selected_population[:]
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(selected_population, 2)
            child = mutate(crossover(parent1, parent2))
            new_population.append(child)
        population = new_population

    # Select the best guess from the final population
    best_guess = min(population, key=fitness)
    return best_guess

def optimal_logical_strategy(previous_guesses, previous_feedbacks):
    """
    Strategy 7: Optimal logical strategy for solving Mastermind.

    Args:
        previous_guesses (list): List of previous guesses.
        previous_feedbacks (list): List of feedbacks for previous guesses.

    Returns:
        list: The next guess based on the optimal logical strategy.
    """
    COLORS = ['RED', 'GREEN', 'BLUE', 'YELLOW', 'ORANGE', 'BLACK']
    all_codes = list(itertools.product(COLORS, repeat=4))

    def get_feedback(secret_code, guess):
        black_pegs = sum(1 for i in range(4) if guess[i] == secret_code[i])
        white_pegs = sum(min(guess.count(color), secret_code.count(color)) for color in set(guess)) - black_pegs
        return black_pegs, white_pegs

    if not previous_guesses:
        # Optimal initial guess for Mastermind (4 columns, 6 colors) is ['RED', 'GREEN', 'BLUE', 'RED']
        return ['RED', 'GREEN', 'BLUE', 'RED']

    # Filter possible codes based on feedback received from previous guesses
    possible_codes = all_codes[:]
    for guess, feedback in zip(previous_guesses, previous_feedbacks):
        possible_codes = [code for code in possible_codes if get_feedback(code, guess) == feedback]

    # Minimax technique to choose the next guess
    def minimize_worst_case(codes):
        min_max_count = float('inf')
        best_guess = None
        for guess in all_codes:
            feedback_counts = {}
            for code in codes:
                feedback = get_feedback(code, guess)
                if feedback not in feedback_counts:
                    feedback_counts[feedback] = 0
                feedback_counts[feedback] += 1
            max_count = max(feedback_counts.values())
            if max_count < min_max_count:
                min_max_count = max_count
                best_guess = guess
        return best_guess

    next_guess = minimize_worst_case(possible_codes)
    return next_guess

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
            guess = random_guess_strategy()
        elif strategy == 1:
            guess = feedback_based_strategy(self.guesses, self.feedbacks)
        elif strategy == 2:
            guess = comprehensive_feedback_strategy(self.guesses, self.feedbacks)
        elif strategy == 3:
            guess = knuth_algorithm_strategy(self.guesses, self.feedbacks)
        elif strategy == 4:
            guess = average_case_strategy(self.guesses, self.feedbacks)
        elif strategy == 5:
            guess = minimax_strategy(self.guesses, self.feedbacks)
        elif strategy == 6:
            guess = genetic_algorithm_strategy(self.guesses, self.feedbacks)
        elif strategy == 7:
            guess = optimal_logical_strategy(self.guesses, self.feedbacks)
        else:
            guess = random_guess_strategy()

        black_pegs, white_pegs = self.get_feedback(guess)
        self.guesses.append(guess)
        self.feedbacks.append((black_pegs, white_pegs))
        self.attempts += 1
        return black_pegs == 4

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
