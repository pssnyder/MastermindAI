import random
"""Mastermind Digital Game

    1. The computer will generate a secret code consisting of a sequence of colors. 
    2. The player will then try to guess the code within a limited number of attempts. 
    3. After each guess, the computer will provide feedback in the form of black and white pegs:
        Black Peg: A color is correct and in the correct position.
        White Peg: A color is correct but in the wrong position.
"""
# Define the colors
COLORS = ['RED', 'GREEN', 'BLUE', 'YELLOW', 'ORANGE', 'BLACK']

# Function to generate a random secret code
def generate_secret_code():
    return [random.choice(COLORS) for _ in range(4)]

# Function to get feedback for a guess
def get_feedback(secret_code, guess):
    black_pegs = 0
    white_pegs = 0
    secret_code_copy = secret_code[:]
    guess_copy = guess[:]

    # First pass: check for correct color and position (black pegs)
    for i in range(4):
        if guess[i] == secret_code[i]:
            black_pegs += 1
            secret_code_copy[i] = None
            guess_copy[i] = None

    # Second pass: check for correct color but wrong position (white pegs)
    for i in range(4):
        if guess_copy[i] is not None and guess_copy[i] in secret_code_copy:
            white_pegs += 1
            secret_code_copy[secret_code_copy.index(guess_copy[i])] = None

    return black_pegs, white_pegs

# Function to print the game board
def print_board(guesses, feedbacks):
    print("\nMastermind Board:")
    print("-----------------")
    for i in range(len(guesses)):
        black_pegs, white_pegs = feedbacks[i]
        print(f"Guess {i+1}: {guesses[i]} -> Feedback: {black_pegs} correct, {white_pegs} correct but wrong position")
    print("-----------------\n")

# Main game function
def play_mastermind():
    secret_code = generate_secret_code()
    attempts = 10
    guesses = []
    feedbacks = []

    print("Welcome to Mastermind!")
    print("Try to guess the secret code consisting of 4 colors.")
    print(f"Available colors: {', '.join(COLORS)}")
    print("You have 10 attempts. Good luck!\n")

    for attempt in range(attempts):
        while True:
            guess = input(f"Attempt {attempt + 1} / {attempts}: Enter your guess (4 colors separated by spaces): ").upper().split()
            if len(guess) == 4 and all(color in COLORS for color in guess):
                break
            else:
                print("Invalid guess. Please enter 4 valid colors separated by spaces.")

        guesses.append(guess)
        black_pegs, white_pegs = get_feedback(secret_code, guess)
        feedbacks.append((black_pegs, white_pegs))
        print_board(guesses, feedbacks)

        if black_pegs == 4:
            print("Congratulations! You've guessed the secret code!")
            break
    else:
        print("Sorry, you've run out of attempts. The secret code was:", secret_code)

# Run the game
if __name__ == "__main__":
    play_mastermind()