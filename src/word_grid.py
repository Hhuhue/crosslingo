from enum import Enum
from logging import Logger

import numpy as np
from tabulate import tabulate

EMPTY_CELL = '-'
BLOCKER_CELL = '■'

class Direction(Enum):
    NONE = 0
    ACROSS = 1
    DOWN = 2
    BLOCKED = 4
    
    @classmethod
    def flip(cls, direction):
        if direction == cls.ACROSS:
            return cls.DOWN
        elif direction == cls.DOWN:
            return cls.ACROSS
        else:
            return direction

class WordGrid:
    def __init__(self, shape: tuple, logger: Logger = None) -> None:
        self.puzzle = np.full(shape, EMPTY_CELL, dtype=np.str_)
        self.shape = np.array(self.puzzle.shape)
        self.state = np.zeros(shape, dtype=np.int8)
        self.logger = logger
        self.flipped = False

    def __str__(self) -> str:
        return str(tabulate(self.puzzle, tablefmt="plain"))
    
    def __repr__(self) -> str:
        return str(self)
    
    def flip(self):
        self.puzzle = self.puzzle.T
        self.shape = self.shape[::-1]
        self.state = self.state.T
        self.flipped = not self.flipped
    
    def reset(self) -> None:
        self.puzzle[:] = EMPTY_CELL
        self.state[:] = 0
        
    def validate_word(self, position: tuple, direction: Direction, word: str) -> bool:
        do_unflip = False
        if direction == Direction.ACROSS and not self.flipped:
            self.flip()
            position = (position[1], position[0])
            do_unflip = True
        
        y, x = position
        is_valid = True
        if len(word) + x > self.puzzle.shape[0]:
            # Word is too long for where it is placed
            if self.logger:
                self.logger.warning(f"Cannot place word of length {len(word)}, '{word}' at {(x, y)}")
            is_valid = False
        elif (self.state[x:x + len(word), y] & direction.value).any():
            # Word is overlapping with an other word in the same direction
            if self.logger:
                self.logger.warning(f"Word overlap detected while trying to place '{word}' at {(x, y)}")
            is_valid = False
        elif x - 1 > 0 and self.puzzle[x - 1, y] != EMPTY_CELL:
            # There is a letter just before the beginning of the word
            if self.logger:
                self.logger.warning(f"Word interference detected while trying to place '{word}' at {(x, y)}")
            is_valid = False
        elif x + len(word) < self.shape[0] and self.puzzle[x + len(word), y] != EMPTY_CELL:
            # There is a letter just after the beginning of the word
            if self.logger:
                self.logger.warning(f"Word interference detected while trying to place '{word}' at {(x, y)}")
            is_valid = False
        
        if do_unflip:
            self.flip()
            
        return is_valid


    def add_word(self, position: tuple, direction: Direction, word: str) -> bool:
        
        if direction == Direction.ACROSS:
            self.flip()
            position = (position[1], position[0])
        
        is_valid = self.validate_word(position, direction, word)
        
        if is_valid:
            y, x = position
            self.puzzle[x:x + len(word), y] = list(word.lower())
            self.state[x:x + len(word), y] |= direction.value
        
        if direction == Direction.ACROSS:
            self.flip()
        
        return is_valid


    def get_letters(self, position: tuple, direction: Direction, length: int):
        y, x = position
        letters = []
        
        if direction == Direction.ACROSS:
            self.flip()
            x, y = position
        else:
            y, x = position
            
        for i, letter in enumerate(self.puzzle[x:x + length, y]):
            if letter == EMPTY_CELL:
                continue
            letters.append((i, letter))
        
        if direction == Direction.ACROSS:
            self.flip()
        
        return letters


    def get_letter(self, position: tuple):
        return self.puzzle[position[1], position[0]]