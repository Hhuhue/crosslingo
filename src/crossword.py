from copy import deepcopy
from enum import Enum
from itertools import product
import random
import sys
from typing import Dict, List, Tuple

from gensim.models import Word2Vec
from loguru import logger
import pandas as pd
from pandas import DataFrame
import numpy as np
from tqdm import tqdm

from words import WordIndex, Word
from word_grid import WordGrid, Direction, ValidationMode

MIN_WORD_LEN = 3

class CrosswordStyle(Enum):
    AMERICAN = 0
    BRITISH = 1
    
class CluesMode(Enum):
    DEFINITION = 0
    SYNONYM = 1
    WORD = 2


class Crossword:
    """Represents a crossword puzzle"""

    def __init__(self, word_grid: WordGrid, words: List[Word], lang_from: str, mode: CluesMode) -> None:
        self.word_grid = word_grid
        self.words = words
        self.lang_from = lang_from
        self.clues = []
        
        fetch_translation = self.words[0].meta.language_code != self.lang_from
        index = WordIndex()
        for word in self.words:
            if fetch_translation:
                trans_words = index.get_translation(word, self.lang_from)
                word = random.choice(trans_words)

            if mode == CluesMode.DEFINITION:
                definitions = index.get_definition(word)
                self.clues.append(random.choice(definitions))
            elif mode == CluesMode.WORD and fetch_translation:
                self.clues.append(word)
            elif mode == CluesMode.SYNONYM:
                synonyms = index.get_synonym(word)
                self.clues.append(random.choice(synonyms))
                
    def to_dict(self):
        return {
            "word_grid": self.word_grid.puzzle.tolist(),
            "words": [str(word) for word in self.words],
            "clues": self.clues
        }
 


class CrosswordGenerator:
    """Crossword generator class"""

    def __init__(
        self, word_index: DataFrame, style: CrosswordStyle, seed: int = 1
    ) -> None:
        """
        Args:
            word_index (DataFrame): Dictionary of all words
            style (CrosswordStyle): Style of crossword
            seed (int, optional): Random seed. Defaults to 1.
        """
        np.random.seed(seed)
        random.seed(seed)
        self.seed = seed
        self.word_index = word_index
        self.snapshots = []
        self.style = style

        # Pre process word index
        self.word_index = self.word_index[~self.word_index.position.isin(['name', 'abbrev', 'symbol'])]
        self.word_index = self.word_index[~self.word_index.word.str.contains(r"[0-9 '-.]")]
        self.word_index = self.word_index[self.word_index.length >= MIN_WORD_LEN]
        self.word_index.loc[:, "frequency"] = self.word_index["frequency"].fillna(1)

    def __get_dictionary(self, lang_code: str, shape: tuple, clues_mode: CluesMode, theme: str = None) -> DataFrame:
        dictionary = self.word_index[self.word_index.language_code == lang_code]
        dictionary = dictionary[dictionary.length <= max(shape)]

        if clues_mode == CluesMode.DEFINITION:
            dictionary = dictionary[dictionary.num_definitions > 0]
        elif clues_mode == CluesMode.SYNONYM:
            dictionary = dictionary[dictionary.num_synonyms > 0]

        if theme:
            model = self.__load_word2vec(lang_code)
            words = model.wv.most_similar([theme], topn=1000)
            rows = []
            for word, sim_score in words:
                row = dictionary[dictionary.word == word]
                row.freq = sim_score
                rows.append(row)
            dictionary = pd.concat(rows)

        if len(dictionary) == 0:
            raise ValueError(
                f"No words found for language {lang_code}"
                + (f" with theme {theme}" if theme else "")
            )

        return dictionary

    def __load_word2vec(self, lang_code: str) -> Word2Vec:
        if lang_code == "de":
            return Word2Vec.load("data/deu_wikipedia_2021_1M/word2vec.model")
        if lang_code == "en":
            return Word2Vec.load("data/eng_wikipedia_2016_1M/word2vec.model")
        if lang_code == "es":
            return Word2Vec.load("data/spa_wikipedia_2021_1M/word2vec.model")
        if lang_code == "fra":
            return Word2Vec.load("data/fra_wikipedia_2021_1M/word2vec.model")

        raise ValueError(f"Unsupported language code {lang_code}")

    def get_steps(self):
        return self.snapshots

    def generate(
        self,
        shape: Tuple[int, int],
        lang_from: str,
        n_words: int,
        lang_to: str = None,
        theme: str = None,
        store_steps: bool = False,
        clues_mode: CluesMode = CluesMode.DEFINITION
    ) -> Crossword:
        """Generates a crossword for the given parameters

        Args:
            shape (Tuple[int, int]): Shape of the puzzle (lines, rows)
            lang_from (str): Language code for the vocabulary to use for clues and words
            n_words (int): Number of words to include in the crossword. (result may contain less)
            lang_to (str, optional): Language code for the words only. Defaults to None.
            theme (str, optional): (Experimental) A theme for the words to use. Defaults to None.
            store_steps (bool, optional): Whether or not to save the crossword after a new word is added. Defaults to False.
            clues_mode (CluesMode, optional): The type of clues to use for the crossword.
        Returns:
            Crossword: A crossword instance with used words and word grid
        """

        self.snapshots = []
        word_grid = WordGrid(shape)
        validation = (
            ValidationMode.SOFT
            if self.style == CrosswordStyle.BRITISH
            else ValidationMode.HARD
        )
        dictionary = self.__get_dictionary(lang_to or lang_from, shape, clues_mode, theme)
        if lang_to and lang_to != lang_from:
            dictionary = dictionary[dictionary.num_translations > 0]

        direction = random.choice([Direction.DOWN, Direction.ACROSS])
        word_list = []

        # List available positions in both directions with words that failed to be placed at each of them
        positions = {
            Direction.DOWN: {
                pos: []
                for pos in product(
                    range(word_grid.shape[1]),
                    range(word_grid.shape[0] - MIN_WORD_LEN + 1),
                )
            },
            Direction.ACROSS: {
                pos: []
                for pos in product(
                    range(word_grid.shape[1] - MIN_WORD_LEN + 1),
                    range(word_grid.shape[0]),
                )
            },
        }

        pbar = tqdm(total=n_words)
        while len(word_list) < n_words:
            # Flip current direction if its positions are exhausted
            if len(positions[direction]) == 0:
                # Exit if all positions are exhausted
                if len(positions[Direction.flip(direction)]) == 0:
                    break
                direction = Direction.flip(direction)

            # Select a random position
            position = random.choice(list(positions[direction]))

            # List potential words for that position
            blacklist = positions[direction][position] + word_list
            candidates = dictionary[
                dictionary["word"].apply(
                    lambda w: word_grid.validate_word(
                        position, direction, w, validation
                    )
                )
            ]
            candidates = candidates[~candidates["word"].isin(blacklist)]

            # Remove position and restart if no candidates
            if len(candidates) == 0:
                positions[direction].pop(position, None)
                continue

            # Chose a word by its frequency and length if possible
            try:
                weights = np.log(np.log(dictionary.freq) + 1) + dictionary.len
                word = candidates.sample(1, weights=weights, random_state=self.seed)
            except Exception:
                word = candidates.sample(1, random_state=12)

            word = Word(word.iloc[0], position, direction)

            # Add the word to the grid
            if not word_grid.add_word(position, direction, word):
                positions[direction][position].append(word)
                logger.opt(lazy=True).debug(f"Can't place word {word} at {position}")
                continue
            word_list.append(word)

            if store_steps:
                self.snapshots.append(
                    (
                        {"position": position, "direction": direction, "word": word},
                        deepcopy(word_grid),
                    )
                )

            # Flip direction if possible
            if len(positions[Direction.flip(direction)]) > 0:
                direction = Direction.flip(direction)

            # Remove newly occupied positions from the list
            if direction == Direction.DOWN:
                used_positions = {
                    (position[0], y)
                    for y in range(position[1] - 1, position[1] + len(word) + 1)
                }
            else:
                used_positions = {
                    (x, position[1])
                    for x in range(position[0] - 1, position[0] + len(word) + 1)
                }

            for pos in used_positions:
                positions[direction].pop(pos, None)

            # Update progress
            pbar.update(1)
            pbar.set_description(
                f"word: {word}, pos: {position}, dir: {direction.name.lower()}, slots {len(positions[Direction.DOWN])}d {len(positions[Direction.ACROSS])}a",
                refresh=True,
            )

        crossword = Crossword(word_grid, word_list, lang_from, clues_mode)
        return crossword


def generate_puzzle_template(shape: tuple, n_words: int) -> WordGrid:
    data = []
    columns = ["word", "freq", "lang_code", "len"]
    for length in range(MIN_WORD_LEN, max(shape)):
        data += [["x" * length, max(shape) - length, "x", length]] * n_words

    word_index = pd.DataFrame(data, columns=columns)

    gen = CrosswordGenerator(
        word_index, CrosswordStyle.BRITISH, random.randint(0, 1000)
    )
    template = gen.generate(shape, "x", n_words)
    print(template.word_grid)


def place_moves(
    moves: List[Word],
    len_groups: Dict[int, DataFrame],
    puzzle: WordGrid,
):
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

        word = moves[current_step]
        if step_words[current_step] is None:
            words = len_groups[len(word)]
            words = words[
                words.word.apply(
                    lambda w: prev_puzzle.validate_word(
                        word.position, word.direction, word
                    )
                )
            ]
            step_words[current_step] = words[~words.word.isin(placed_words)]

        if len(step_words[current_step]) == 0:
            if current_step == 0:
                return best_step
            step_words[current_step] = None
            current_step -= 1
            pbar.update(-1)
            continue

        while len(step_words[current_step]) > 0:
            word = Word(
                step_words[current_step].sample(1), word.position, word.direction
            )
            step_words[current_step].drop(word.meta.index, inplace=True)

            step_puzzle = deepcopy(prev_puzzle)
            if step_puzzle.add_word(word.position, word.direction, word):
                step_puzzles[current_step] = step_puzzle
                current_step_word = placed_words[current_step]
                placed_words[current_step] = word
                if not current_step_word:
                    best_step = (deepcopy(placed_words), step_puzzle)
                current_step += 1
                pbar.update(1)
                break

    return best_step


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, level="ERROR")

    word_index = WordIndex().get_data()
    gen = CrosswordGenerator(word_index, CrosswordStyle.BRITISH, seed=18)
    crossword = gen.generate((8, 16), "en", 20, lang_to="de", theme="cat")
    print(crossword.words)
    print(crossword.word_grid)

    for clue in crossword.clues:
        print(clue)
