"""
Information Theory Lab

Benjamin McMullin
Math 403
9/26/2023
"""


import numpy as np
import wordle

# Problem 1
def get_guess_result(guess, true_word):
    """
    Returns an array containing the result of a guess, with the return values as follows:
        2 - correct location of the letter
        1 - incorrect location but present in word
        0 - not present in word
    For example, if the secret word is "boxed" and the provided guess is "excel", the 
    function should return [0,1,0,2,0].
    
    Arguments:
        guess (string) - the guess being made
        true_word (string) - the secret word
    Returns:
        result (list of integers) - the result of the guess, as described above
    """
    # initialize the return list and the length of the word
    length = 5
    colors = [None] * length
    # turn the true word into a list
    secret = list(true_word)

    # create a dictionary of the letters in the guess and their positions
    letters = {}
    for i, let in enumerate(guess):
        if let not in letters:
            letters[let] = []
        letters[let].append(i)

    index = -1
    for let in letters:
        update = 0

        # check each position of the letter
        while len(letters[let]) > 0:
            color = 0
            failed = False
            for pos in letters[let]:
                # if the letter is not in the secret word, break and move on, indicating failure to prevent updating the secrete word
                try:
                    index = secret.index(guess[pos])

                # if the letter is not in the secret word, break and move on, indicating failure to prevent updating the secrete word
                except ValueError:
                    color = 0
                    update = pos
                    failed = True
                    break

                # if the letter is in the secret word, check if it is in the correct position
                if guess[pos] == secret[pos]:
                    color = 2
                    update = pos
                    break

                # if the letter is in the secret word, but not in the correct position, check if it is in the wrong position
                else:
                    if 1 > color:
                        color = 1
                        update = pos

            # if the letter is in the secret word, but not in the correct position, check if it is in the wrong position
            if not failed:
                secret[index] = "_"

            # update the return list and the dictionary
            colors[update] = color
            letters[let].remove(update)

    return colors


# Helper function
def load_words(filen):
    """
    Loads all of the words from the given file, ensuring that they 
    are formatted correctly.
    """
    with open(filen, 'r') as file:
        # Get all 5-letter words
        words = [line.strip() for line in file.readlines() if len(line.strip()) == 5]
    return words
    
# Problem 2
def compute_highest_entropy(all_guess_results, allowed_guesses):
    """
    Compute the entropy of each allowed guess.
    
    Arguments:
        all_guess_results ((n,m) ndarray) - the array found in
            all_guess_results.npy, containing the results of each 
            guess for each secret word, where n is the the number
            of allowed guesses and m is number of possible secret words.
        allowed_guesses (list of strings) - list of the allowed guesses
    Returns:
        (string) The highest-entropy guess
    """
    # Get the indices of the allowed guesses
    entropy = - np.array([np.sum(
        count / np.sum(count) * np.log2(count / np.sum(count))
    )
        # Get the counts of each result
        for _, count in [np.unique(row, return_counts=True)
                          for row in all_guess_results]
    ])

    # Return the guess with the highest entropy
    return allowed_guesses[np.argmax(entropy)]
    
# Problem 3
def filter_words(all_guess_results, allowed_guesses, possible_secret_words, guess, result):
    """
    Create a function that filters the list of possible words after making a guess.
    Since we already have an array of the result of all guesses for all possible words, 
    we will use this array instead of recomputing the results.
    
	Return a filtered list of possible words that are still possible after 
    knowing the result of a guess. Also return a filtered version of the array
    of all guess results that only contains the results for the secret words 
    still possible after making the guess. This array will be used to compute 
    the entropies for making the next guess.
    
    Arguments:
        all_guess_results (2-D ndarray)
            The array found in all_guess_results.npy, 
            containing the result of making any allowed 
            guess for any possible secret word
        allowed_guesses (list of str)
            The list of words we are allowed to guess
        possible_secret_words (list of str)
            The list of possible secret words
        guess (str)
            The guess we made
        result (tuple of int)
            The result of the guess
    Returns:
        (list of str) The filtered list of possible secret words
        (2-D ndarray) The filtered array of guess results
    """
    # Get the indices of the allowed guesses and the guess
    guess = allowed_guesses.index(guess)

    # Convert the result to base 3
    power = np.array([3**i for i in range(5)])
    base3 = np.sum(result*power)

    # Get the indices of the matches
    match = np.argwhere(all_guess_results == base3)
    match = [row[1] for row in match if row[0] == guess]

    # Return the filtered list of possible words and the filtered array of guess results
    return list(np.array(possible_secret_words)[match]), all_guess_results[:, match]

# Problem 4
def play_game_naive(game, all_guess_results, possible_secret_words, allowed_guesses, word=None, display=False):
    """
    Plays a game of Wordle using the strategy of making guesses at random.
    
    Return how many guesses were used.
    
    Arguments:
        game (wordle.WordleGame)
            the Wordle game object
        all_guess_results ((n,m) ndarray)
            The array found in all_guess_results.npy, 
            containing the result of making any allowed 
            guess for any possible secret word
        possible_secret_words (list of str)
            list of possible secret words
        allowed_guesses (list of str)
            list of allowed guesses
        
        word (optional)
            If not None, this is the secret word; can be used for testing. 
        display (bool)
            If true, output will be printed to the terminal by the game.
    Returns:
        (int) Number of guesses made
    """
    # Initialize the game
    game.start_game(word=word, display=display)
    count = 0
    
    while not game.is_finished():
        if len(possible_secret_words) == 1:
            # Choose the last possible word
            naive_guess = possible_secret_words[0]
        else:
            # Choose a random word
            naive_guess = np.random.choice(allowed_guesses)

        # Make the guess and update game, possible words, and results
        naive_result, count = game.make_guess(naive_guess)
        possible_secret_words, all_guess_results = filter_words(all_guess_results, allowed_guesses, possible_secret_words,
                                                         naive_guess, naive_result)
    return count

# Problem 5
def play_game_entropy(game, all_guess_results, possible_secret_words, allowed_guesses, word=None, display=False):
    """
    Plays a game of Wordle using the strategy of guessing the maximum-entropy guess.
    
    Return how many guesses were used.
    
    Arguments:
        game (wordle.WordleGame)
            the Wordle game object
        all_guess_results ((n,m) ndarray)
            The array found in all_guess_results.npy, 
            containing the result of making any allowed 
            guess for any possible secret word
        possible_secret_words (list of str)
            list of possible secret words
        allowed_guesses (list of str)
            list of allowed guesses
        
        word (optional)
            If not None, this is the secret word; can be used for testing. 
        display (bool)
            If true, output will be printed to the terminal by the game.
    Returns:
        (int) Number of guesses made
    """
    # Initialize the game
    game.start_game(word=word, display=display)
    
    count = 0

    while not game.is_finished():
        if len(possible_secret_words) == 1:
            # Choose the last possible word
            entropy_guess = possible_secret_words[0]
        else:
            # Choose the word with the highest entropy
            entropy_guess = compute_highest_entropy(all_guess_results, allowed_guesses)

        # Make the guess and update game, possible words, and results
        result, count = game.make_guess(entropy_guess)
        possible_secret_words, all_guess_results = filter_words(all_guess_results, allowed_guesses,
                                                                possible_secret_words, entropy_guess, result)

    return count

# Problem 6
def compare_algorithms(all_guess_results, possible_secret_words, allowed_guesses, n=20):
    """
    Compare the algorithms created in Problems 5 and 6. Play n games with each
    algorithm. Return the mean number of guesses the algorithms from
    problems 5 and 6 needed to guess the secret word, in that order.
    
    
    Arguments:
        all_guess_results ((n,m) ndarray)
            The array found in all_guess_results.npy, 
            containing the result of making any allowed 
            guess for any possible secret word
        possible_secret_words (list of str)
            list of possible secret words
        allowed_guesses (list of str)
            list of allowed guesses
        n (int)
            Number of games to run
    Returns:
        (float) - average number of guesses needed by naive algorithm
        (float) - average number of guesses needed by entropy algorithm
    """
    # Initialize the scores
    naive_scores = [play_game_naive(wordle.WordleGame(), all_guess_results, possible_secret_words, allowed_guesses)
                    for _ in range(n)]
    entropy_scores = [play_game_entropy(wordle.WordleGame(), all_guess_results, possible_secret_words, allowed_guesses)
                      for _ in range(n)]

    # Return the mean of the scores
    return np.mean(np.array(naive_scores)), np.mean(np.array(entropy_scores))
    
if __name__ == "__main__":
    pass
 # # test problem 1
    # true_words = ["boxed", "apple", "banan", "happy", "eeeee", "eeeee", "asdfg", "asdfg", "aback", "aahed"]
    # guesses =    ["excel", "orang", "banan", "hobby", "abcde", "ebcbe", "sdfga", "qwert", "aahed", "aback"]
    # # true_words = ["eeeee"]
    # # guesses = ["abcde"]
    # expected_results = [
    #     [0, 1, 0, 2, 0],
    #     [0, 0, 1, 0, 0],
    #     [2, 2, 2, 2, 2],
    #     [2, 0, 0, 0, 2],
    #     [0, 0, 0, 0, 2],
    #     [2, 0, 0, 0, 2],
    #     [1, 1, 1, 1, 1],
    #     [0, 0, 0, 0, 0],
    #     [2, 1, 0, 0, 0],
    #     [2, 0, 1, 0, 0],
    # ]
    
    # # Test the function using a for loop
    # for i in range(len(true_words)):
    #     true_word = true_words[i]
    #     guess = guesses[i]
    #     expected_result = expected_results[i]
    #     guessed = get_guess_result(guess, true_word)
    #     success = guessed == expected_result
    
    #     print(f"Test {i}:", success)
    #     if not success:
    #         print("\tTruth:\t", true_word)
    #         print("\tGuess:\t", guess)
    #         print("\tReturned:\t", guessed)
    #         print("\tExpected:\t",expected_result)

    # # test old problem 2
    # secrets = load_words("possible_words.txt")
    # allowed = load_words("allowed_words.txt")
    
    # np.save("all_guess_results.npy", get_all_guess_results(secrets, allowed))

    # # test problem 2
    # words = np.load("all_guess_results.npy")
    # test_words = words.copy()
    
    # results = compute_highest_entropy(test_words, load_words("allowed_guesses.txt"))
    # print(results)

    # test problem 3
    # matrix = np.load("all_guess_results.npy").copy()
    # guess, index = compute_highest_entropy(matrix, load_words("allowed_guesses.txt"))
    # secret = "heart"
    # filter_words(matrix, load_words("possible_secret_words.txt"), index, get_guess_result(secret, guess))


    # # test problem 4
    # game = wordle.WordleGame()
    # matrix = np.load("all_guess_results.npy").copy()
    # possible = load_words("possible_secret_words.txt")
    # allowed = load_words("allowed_guesses.txt")
    # play_game_naive(game, matrix, possible, allowed, display=True)

    # test problem 5
    # game = wordle.WordleGame()
    # matrix = np.load("all_guess_results.npy").copy()
    # possible = load_words("possible_secret_words.txt")
    # allowed = load_words("allowed_guesses.txt")
    # play_game_entropy(game, matrix, possible, allowed, display=True)

    # test problem 6
    # matrix = np.load("all_guess_results.npy").copy()
    # possible = load_words("possible_secret_words.txt")
    # allowed = load_words("allowed_guesses.txt")
    # print(compare_algorithms(matrix, possible, allowed))





    '''Problem 1'''
    # assert get_guess_result("excel", "boxed") == [0,1,0,2,0], 'Wrong answer for "excel" and "boxed"'
    # assert get_guess_result("stare", "train") == [0, 1, 2, 1, 0], 'Wrong answer for "stare" and "train"'
    # assert get_guess_result("green", "pages") == [1, 0, 0, 2, 0], 'Wrong answer for "green" and "pages"'
    # assert get_guess_result("abate", "vials") == [0, 0, 2, 0, 0], 'Wrong answer for "abate" and "vials"'
    # assert get_guess_result("robot", "older") == [1, 1, 0, 0, 0], 'Wrong answer for "robot" and "older"'

    '''Problem 2'''
    # file = np.load('all_guess_results.npy').copy()
    # guess_results = np.load('all_guess_results.npy', allow_pickle=True)
    # allowed = load_words('allowed_guesses.txt')
    # print(compute_highest_entropy(guess_results, allowed))
    # # Should give us the word 'soare'.

    '''Problem 3'''
    # all_guess_results = np.load('all_guess_results.npy')
    # possible_secret_words = load_words('possible_secret_words.txt')
    # allowed_guesses = load_words('allowed_guesses.txt')
    # result = [0, 0, 0, 2, 1]
    # guess = 'boxes'
    # new_possible_words, new_guess_results = filter_words(all_guess_results, allowed_guesses, possible_secret_words, guess, result)
    # print("New possible words:", len(new_possible_words))
    # print("New possible words:", new_possible_words)
    # print("New guess results:", new_guess_results.shape)

    '''Problem 4'''
    # game = wordle.WordleGame()
    # all_guess_results = np.load('all_guess_results.npy')
    # possible_secret_words = load_words('possible_secret_words.txt')
    # allowed_guesses = load_words('allowed_guesses.txt')
    # play_game_naive(game, all_guess_results, possible_secret_words, allowed_guesses, word='booze', display=True)

    '''Problem 5'''
    '''
    TODO:
    Check and make sure I am returning the right count. Should it be the one before the guess, or which count
    the guess should be (so if it takes 3 guesses to get there, should it be 2 [when the list of allowed_words is 1,
    or should it be 3 when we would make the actual guess?])
    '''
    # game = wordle.WordleGame()
    # all_guess_results = np.load('all_guess_results.npy')
    # possible_secret_words = load_words('possible_secret_words.txt')
    # allowed_guesses = load_words('allowed_guesses.txt')
    # print(play_game_entropy(game, all_guess_results, possible_secret_words, allowed_guesses, word='booze', display=True))

    '''Problem 6'''
    # game = wordle.WordleGame()
    # all_guess_results = np.load('all_guess_results.npy')
    # possible_secret_words = load_words('possible_secret_words.txt')
    # allowed_guesses = load_words('allowed_guesses.txt')
    # print(compare_algorithms(all_guess_results, possible_secret_words, allowed_guesses, n=20))