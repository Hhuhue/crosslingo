from itertools import product
import logging
import random

from gensim.models import Word2Vec
import pandas as pd
from pandas import DataFrame
import numpy as np
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
        candidates = get_candidates(puzzle, position, direction, blacklist)

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

            positions[direction].pop(position, None)    
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