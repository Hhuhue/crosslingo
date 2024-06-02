from enum import Enum
from itertools import product
import random

import pandas as pd
from pandas import DataFrame
import numpy as np

MIN_WORD_LEN = 3

class Direction(Enum):
    ACROSS = 0
    DOWN = 1

class CrosswordGenerator:
    def __init__(self) -> None:
        self.word_index = pd.read_csv("data/word_index.csv")
        self.puzzle: np.ndarray = None
        self.valid_pos: dict = None

    def place_word(self, word: str, position: tuple, direction: Direction, valid_pos: dict):
        if direction == Direction.ACROSS:
            self.puzzle[position[0], position[1]:] = list(word)
            valid_pos[direction].discard((position[0], position[1] - 1))
            valid_pos[direction].discard((position[0], position[1] + len(word) - 1))
        else:
            self.puzzle[position[0], position[1]:] = list(word)
            valid_pos[direction].discard((position[0] - 1, position[1]))
            valid_pos[direction].discard((position[0] + len(word) - 1, position[1]))

    def generate(self, shape: tuple, lang: str, n: int):
        dictionary = self.word_index[self.word_index["lang_code"] == lang]
        dictionary = dictionary[dictionary["len"] >= MIN_WORD_LEN]
        
        positions = list(product(range(shape[0] - MIN_WORD_LEN + 1), range(shape[1] - MIN_WORD_LEN + 1)))
        valid_pos = {
            Direction.ACROSS: set(positions),
            Direction.DOWN: set(positions)
        }
        
        self.puzzle = np.chararray(shape)
        self.puzzle[:] = ' '
        
        words_placed = 0
        while words_placed < n:
            direction = random.choice(list(Direction))
            position = random.choice(valid_pos[direction])
            
            valid_lengths, chars_to_include = self.get_constraints(shape, direction, position)       
            candidates = dictionary[dictionary["len"].isin(valid_lengths)]
            candidates = candidates[candidates["word"].apply(lambda w: all([w[i] == v for i, v in chars_to_include.items()]))]
            
            if len(candidates) == 0:
                valid_pos[direction].discard(position)
                continue
            
            word = candidates["word"].sample(1).to_list()[0]
            
            


    def get_constraints(self, shape: tuple, direction: Direction, position: tuple):
        valid_lengths = []
        chars_to_include = {}
        if direction == Direction.ACROSS:
            for i in range(position[0], shape[0]):
                if self.puzzle[position[0] + i, position[1]] != '':
                    chars_to_include[i] = self.puzzle[position[0] + i, position[1]]
                if i > MIN_WORD_LEN - 1 and (position[0] + i < shape[0] or self.puzzle[position[0] + i, position[1]] == ''):
                    valid_lengths.append(i)
        else:
            for i in range(position[1], shape[1]):
                if self.puzzle[position[0], position[1] + i] != '':
                    chars_to_include[i] = self.puzzle[position[0], position[1] + i]
                if i > MIN_WORD_LEN - 1 and (position[1] + i < shape[1] or self.puzzle[position[0], position[1] + i] == ''):
                    valid_lengths.append(i)
                    
        return valid_lengths, chars_to_include