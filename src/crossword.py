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

from word_grid import WordGrid, Direction, ValidationMode

MIN_WORD_LEN = 3


class Word(str):
    """Represents a crossword word placement"""

    def __new__(cls, word: pd.Series, position: Tuple[int, int], direction: Direction):
        obj = str.__new__(cls, word.word.item())
        obj.meta = word
        obj.direction = direction
        obj.position = position
        return obj


class CrosswordStyle(Enum):
    AMERICAN = 0
    BRITISH = 1


class Crossword:
    """Represents a crossword puzzle"""

    def __init__(self, word_grid: WordGrid, words: List[Word], lang_from: str) -> None:
        self.word_grid = word_grid
        self.words = words
        self.lang_from = lang_from


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

    def __get_dictionary(self, lang_code: str, theme: str = None) -> DataFrame:
        dictionary = self.word_index[self.word_index["lang_code"] == lang_code]
        dictionary.freq.fillna(1, inplace=True)
        dictionary.loc[:, "word"] = dictionary["word"].astype(str)
        dictionary = dictionary[dictionary["len"] >= MIN_WORD_LEN]
        dictionary = dictionary[~dictionary["word"].str.contains(r"[0-9 '-]")]

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
                f"No words found for language {lang_code}" + f" with theme {theme}"
                if theme
                else ""
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
    ) -> Crossword:
        """Generates a crossword for the given parameters

        Args:
            shape (Tuple[int, int]): Shape of the puzzle (lines, rows)
            lang_from (str): Language code for the vocabulary to use for clues and words
            n_words (int): Number of words to include in the crossword. (result may contain less)
            lang_to (str, optional): Language code for the words only. Defaults to None.
            theme (str, optional): (Experimental) A theme for the words to use. Defaults to None.
            store_steps (bool, optional): Whether or not to save the crossword after a new word is added. Defaults to False.

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
        dictionary = self.__get_dictionary(lang_to or lang_from, theme)
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

            word = Word(word, position, direction)

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
                f"word: {word}, pos: {position}, dir: {direction.name.lower()}, cnd: {len(candidates)}, slots {len(positions[Direction.DOWN])}d {len(positions[Direction.ACROSS])}a",
                refresh=True,
            )

        return Crossword(word_grid, word_list, lang_from)


def generate_puzzle_template(shape: tuple, n_words: int) -> WordGrid:
    data = []
    columns = ["word", "freq", "lang_code", "len"]
    for length in range(MIN_WORD_LEN, max(shape)):
        data += [["x" * length, max(shape) - length, "x", length]] * n_words

    word_index = pd.DataFrame(data, columns=columns)

    gen = CrosswordGenerator(word_index, CrosswordStyle.BRITISH, random.randint(0, 1000))
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

    word_index = pd.read_csv("data/word_index.csv", encoding="utf-8")
    gen = CrosswordGenerator(word_index, CrosswordStyle.BRITISH, seed=23)
    crossword = gen.generate((10, 20), "en", 25)
    print(crossword.words)
    print(crossword.word_grid)
