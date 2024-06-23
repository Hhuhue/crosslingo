from itertools import product
import logging
import random

import pandas as pd
from pandas import DataFrame
from tqdm import tqdm

from word_grid import WordGrid, Direction

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


def load_dictionary(shape: tuple) -> dict:
    word_index = pd.read_csv("data/word_index.csv", encoding='utf-8')
    dictionary = word_index[word_index["lang_code"] == "en"]
    dictionary["word"] = dictionary["word"].astype(str)
    dictionary = dictionary[dictionary["len"] >= MIN_WORD_LEN]
    dictionary = dictionary[dictionary["len"] <= max(shape)]
    dictionary = dictionary[~dictionary["word"].str.contains(r"[0-9]")]
    print(len(dictionary))
    return dictionary
        

def get_candidates(puzzle: WordGrid, dictionary: DataFrame, position: tuple, direction: Direction, blacklist: list) -> DataFrame:
    if direction == Direction.ACROSS:
        x, y = position
        max_len = puzzle.shape[1] - position[0]
        puzzle.flip()
    else:
        max_len = puzzle.shape[0] - position[1]
        y, x = position

    candidates = dictionary[dictionary["word"].apply(lambda w: puzzle.validate_word(x, y, direction, w))]
    
    if direction == Direction.ACROSS:
        puzzle.flip()
    
    candidates = candidates[~candidates["word"].isin(blacklist)]
        
    letters = puzzle.get_letters(position, direction, max_len)
    for index, letter in letters:
        candidates = candidates[candidates["word"].str[index] == letter]
        
    return candidates


def create_crossword(puzzle: WordGrid, dictionary: DataFrame):
    n = 0
    direction = Direction.DOWN
    word_list = []
    puzzle.reset()
    positions = {
        Direction.DOWN: {pos: [] for pos in product(range(puzzle.shape[1]), range(puzzle.shape[0] - MIN_WORD_LEN))},
        Direction.ACROSS: {pos: [] for pos in product(range(puzzle.shape[1] - MIN_WORD_LEN), range(puzzle.shape[0]))}
    }

    pbar = tqdm()
    while n < 12:
        if len(positions[direction]) == 0:
            if len(positions[Direction.flip(direction)]) == 0:
                break
            direction = Direction.flip(direction)
        
        position = random.choice(list(positions[direction]))
        
        blacklist = positions[direction][position] + word_list
        candidates = get_candidates(puzzle, dictionary, position, direction, blacklist)

        if len(candidates) == 0:
            positions[direction].pop(position, None)
            continue
        
        try:
            word = candidates["word"].sample(1, weights=candidates.freq).item()
        except:
            word = candidates["word"].sample(1).item()
        
        pbar.update(n)
        #pbar.set_description(f"word: {word}, pos: {position}, dir: {direction.name.lower()}, cnd: {len(candidates)}, slots {len(positions[Direction.DOWN])}", refresh=True)
        print(f"word: {word}, pos: {position}, dir: {direction.name.lower()}, cnd: {len(candidates)}, slots {len(positions[Direction.DOWN])}d {len(positions[Direction.ACROSS])}a")
        
        if puzzle.add_word(position, direction, word):
            if len(positions[Direction.flip(direction)]) > 0:
                direction = Direction.flip(direction)

            positions[direction].pop(position, None)    
            word_list.append(word)
            n += 1
            print(puzzle)
        else:
            positions[direction][position].append(word)

    print(word_list)
    print(puzzle)
    
if __name__ == "__main__":
    logger = get_logger()
    puzzle = WordGrid((5, 10), logger, False)
    dictionary = load_dictionary(puzzle.shape)
    create_crossword(puzzle)