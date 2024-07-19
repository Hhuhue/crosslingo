from copy import deepcopy
from itertools import product
import logging
import random
from typing import Dict, List, Tuple

from gensim.models import Word2Vec
import pandas as pd
from pandas import DataFrame
import numpy as np
from tqdm import tqdm

from word_grid import WordGrid, Direction,ValidationMode

MIN_WORD_LEN = 3

def get_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.ERROR)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.ERROR)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger

def load_dictionary() -> dict:
    word_index = pd.read_csv("data/word_index.csv", encoding='utf-8')
    dictionary = word_index[word_index["lang_code"] == "en"]
    dictionary["word"] = dictionary["word"].astype(str)
    dictionary = dictionary[dictionary["len"] >= MIN_WORD_LEN]
    dictionary = dictionary[~dictionary["word"].str.contains(r"[0-9 '-]")]
    print(len(dictionary))
    return dictionary

def load_word2vec(lang_code: str) -> Word2Vec:
    if lang_code == "de":
        return Word2Vec.load("data/deu_wikipedia_2021_1M/word2vec.model")
    elif lang_code == "en":
        return Word2Vec.load("data/eng_wikipedia_2016_1M/word2vec.model")
    elif lang_code == "es":
        return Word2Vec.load("data/spa_wikipedia_2021_1M/word2vec.model")
    elif lang_code == "fra":
        return Word2Vec.load("data/fra_wikipedia_2021_1M/word2vec.model")
    
def generate_puzzle_template(shape: tuple, n_words: int) -> WordGrid:
    puzzle = WordGrid(shape)

    direction = Direction.DOWN
    moves = []
    positions = {
        Direction.DOWN: {pos for pos in product(range(puzzle.shape[1]), range(puzzle.shape[0] - MIN_WORD_LEN + 1))},
        Direction.ACROSS: {pos for pos in product(range(puzzle.shape[1] - MIN_WORD_LEN + 1), range(puzzle.shape[0]))}
    }
    pbar = tqdm(total=n_words)
    while len(moves) < n_words:
        if len(positions[direction]) == 0:
            if len(positions[Direction.flip(direction)]) == 0:
                break
            direction = Direction.flip(direction)
        
        position = random.choice(list(positions[direction]))
        if direction == Direction.DOWN:
            max_length = puzzle.shape[0] - position[1]
        else:
            max_length = puzzle.shape[1] - position[0]
            

        if len(moves) > 0:
            letters = puzzle.get_letters(position, direction, max_length)
            if len(letters) == 0:                
                positions[direction].remove(position)
                continue
            min_length = max(MIN_WORD_LEN, letters[0][0] + 1)
        else:
            position = (position[0], 0)
            min_length = MIN_WORD_LEN
            
        length = np.random.randint(min_length, max_length) if min_length != max_length else min_length
        if not puzzle.add_word(position, direction, "x" * length):
            positions[direction].remove(position)
            continue
            
        moves.append((position, length, direction))
        
        if direction == Direction.DOWN:
            used_positions = {(position[0], y) for y in range(position[1] - 1, position[1] + length + 1)}
        else:
            used_positions = {(x, position[1]) for x in range(position[0] - 1, position[0] + length + 1)}
            
        positions[direction].difference_update(used_positions)
        direction = Direction.flip(direction)
        pbar.update(1)
    puzzle.color_print()
    return moves 

def place_moves(moves: List[Tuple[tuple, int, Direction]], len_groups: Dict[int, DataFrame], puzzle: WordGrid):
    current_step = 0
    placed_words = ["" for _ in range(len(moves))]
    step_puzzles = [None for _ in range(len(moves))]
    step_words = [None for _ in range(len(moves))]
    best_step = None
    
    pbar = tqdm(total=len(moves))
    while current_step < len(moves):
        if current_step == 0:
            prev_puzzle = puzzle
        else:
            prev_puzzle = step_puzzles[current_step - 1]

        position, length, direction = moves[current_step]
        if step_words[current_step] is None:
            words = len_groups[length]
            words = words[words.word.apply(lambda w: prev_puzzle.validate_word(position, direction, w))]
            step_words[current_step] = words[~words.word.isin(placed_words)]
    
        if len(step_words[current_step]) == 0:
            if current_step == 0:
                return best_step
            step_words[current_step] = None
            current_step -= 1
            pbar.update(-1)
            continue
    
        while len(step_words[current_step]) > 0:
            word = step_words[current_step].sample(1).word
            step_words[current_step].drop(word.index, inplace=True)
            word = word.item()
            step_puzzle = deepcopy(prev_puzzle)
            if step_puzzle.add_word(position, direction, word):
                step_puzzles[current_step] = step_puzzle
                current_step_word = placed_words[current_step]
                placed_words[current_step] = word
                if not current_step_word:
                    best_step = (deepcopy(placed_words), step_puzzle)
                current_step += 1
                pbar.update(1)
                break
    
    return best_step

def create_crossword(puzzle: WordGrid, dictionary: DataFrame, n_word: int):
    seed = random.randint(0, 1000)

    random.seed(seed)
    direction = Direction.DOWN
    word_list = []
    #snapshots = []
    positions = {
        Direction.DOWN: {pos: [] for pos in product(range(puzzle.shape[1]), range(puzzle.shape[0] - MIN_WORD_LEN + 1))},
        Direction.ACROSS: {pos: [] for pos in product(range(puzzle.shape[1] - MIN_WORD_LEN + 1), range(puzzle.shape[0]))}
    }
    pbar = tqdm(total=n_word)
    while len(word_list) < n_word:
        if len(positions[direction]) == 0:
            if len(positions[Direction.flip(direction)]) == 0:
                break
            direction = Direction.flip(direction)
        
        position = random.choice(list(positions[direction]))
        
        blacklist = positions[direction][position] + word_list
        candidates = dictionary[dictionary["word"].apply(lambda w: puzzle.validate_word(position, direction, w, ValidationMode.HARD))]
        candidates = candidates[~candidates["word"].isin(blacklist)]

        if len(candidates) == 0:
            positions[direction].pop(position, None)
            continue
        
        try:
            weights = np.log(np.log(dictionary.freq.fillna(1)) + 1) + dictionary.len
            word = candidates.word.sample(1, weights=weights, random_state=seed).item()
        except:
            word = candidates.word.sample(1, random_state=12).item()
        
        pbar.update(1)
        pbar.set_description(f"word: {word}, pos: {position}, dir: {direction.name.lower()}, cnd: {len(candidates)}, slots {len(positions[Direction.DOWN])}d {len(positions[Direction.ACROSS])}a", refresh=True)
        
        if puzzle.add_word(position, direction, word):
            #snapshots.append(({"position": position, "direction": direction, "word": word}, deepcopy(puzzle)))
            if len(positions[Direction.flip(direction)]) > 0:
                direction = Direction.flip(direction)
            
            if direction == Direction.DOWN:
                used_positions = {(position[0], y) for y in range(position[1] - 1, position[1] + len(word) + 1)}
            else:
                used_positions = {(x, position[1]) for x in range(position[0] - 1, position[0] + len(word) + 1)}
                
            for pos in used_positions:
                positions[direction].pop(pos, None)
            
            word_list.append(word)
        else:
            positions[direction][position].append(word)
            logger.info(f"Can't place word {word} at {position}")

    print(word_list)
    print(puzzle)
    
if __name__ == "__main__":
    logger = get_logger()
    puzzle = WordGrid((5, 10))
    dictionary = load_dictionary()
    create_crossword(puzzle, dictionary, 15)