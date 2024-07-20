from enum import Enum

from loguru import logger
import numpy as np
from tabulate import tabulate
from termcolor import colored

EMPTY_CELL = "-"
BLOCKER_CELL = "■"


class Direction(Enum):
    NONE = 0
    ACROSS = 1
    DOWN = 2
    BLOCKED = 4

    @classmethod
    def flip(cls, direction):
        if direction == cls.ACROSS:
            return cls.DOWN
        if direction == cls.DOWN:
            return cls.ACROSS
        
        return direction


class ValidationMode:
    SOFT = 0
    HARD = 1


class WordGrid:
    def __init__(self, shape: tuple) -> None:
        self.puzzle = np.full(shape, EMPTY_CELL, dtype=np.str_)
        self.shape = np.array(self.puzzle.shape)
        self.state = np.zeros(shape, dtype=np.int8)
        self.flipped = False

    def __str__(self) -> str:
        return str(tabulate(self.puzzle, tablefmt="plain"))

    def __repr__(self) -> str:
        return str(self)

    def __validate_word_sides(
        self, x: int, y: int, direction: Direction, word: str, mode: ValidationMode
    ) -> bool:
        word_region = self.state[x : x + len(word), :]
        word_sides = [y + o for o in [-1, 1] if 0 <= y + o < self.shape[1]]
        side_letters = word_region[word_region[:, y] == Direction.NONE.value][
            :, word_sides
        ]

        if mode == ValidationMode.SOFT:
            return not (side_letters & Direction.flip(direction).value).any()
        else:
            return (side_letters == Direction.NONE.value).all()

    def color_print(self) -> None:
        to_print = []
        for chars, states in zip(self.puzzle, self.state):
            data = []
            for char, state in zip(chars, states):
                if state & Direction.ACROSS.value and state & Direction.DOWN.value:
                    color = "magenta"
                elif state & Direction.ACROSS.value:
                    color = "blue"
                elif state & Direction.DOWN.value:
                    color = "yellow"
                else:
                    color = "grey"
                data.append(colored(char, color))
            to_print.append(data)
        print(tabulate(to_print))

    def flip(self):
        self.puzzle = self.puzzle.T
        self.shape = self.shape[::-1]
        self.state = self.state.T
        self.flipped = not self.flipped

    def reset(self) -> None:
        self.puzzle[:] = EMPTY_CELL
        self.state[:] = 0

    def validate_word(
        self,
        position: tuple,
        direction: Direction,
        word: str,
        mode: ValidationMode = ValidationMode.SOFT,
    ) -> bool:
        do_unflip = False
        if direction == Direction.ACROSS and not self.flipped:
            self.flip()
            position = (position[1], position[0])
            do_unflip = True

        y, x = position
        is_valid = True
        if len(word) + x > self.puzzle.shape[0]:
            # Word is too long for where it is placed
            logger.opt(lazy=True).debug(
                f"Cannot place word of length {len(word)}, '{word}' at {(x, y)}"
            )
            is_valid = False
        elif (self.state[x : x + len(word), y] & direction.value).any():
            # Word is overlapping with an other word in the same direction
            logger.opt(lazy=True).debug(
                f"Word overlap detected while trying to place '{word}' at {(x, y)}"
            )
            is_valid = False
        elif x - 1 >= 0 and self.puzzle[x - 1, y] != EMPTY_CELL:
            # There is a letter just before the beginning of the word
            logger.opt(lazy=True).debug(
                f"Start of word interference detected while trying to place '{word}' at {(x, y)}"
            )
            is_valid = False
        elif (
            x + len(word) < self.shape[0]
            and self.puzzle[x + len(word), y] != EMPTY_CELL
        ):
            # There is a letter just after the end of the word
            logger.opt(lazy=True).debug(
                f"End of word interference detected while trying to place '{word}' at {(x, y)}"
            )
            is_valid = False
        elif any(map(lambda il: word[il[0]] != il[1], self.get_letters(position, direction, len(word)))):
            # Make sure the word doesn't replace letters already present
            logger.opt(lazy=True).debug(
                f"Letter conflict detected while trying to place '{word}' at {(x, y)}"
            )
            is_valid = False
        elif not self.__validate_word_sides(x, y, direction, word, mode):
            # Make sure the word isn't touching another word above or below in opposite direction
            logger.opt(lazy=True).debug(
                f"Side of word interference detected while trying to place '{word}' at {(x, y)}"
            )
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
            self.puzzle[x : x + len(word), y] = list(word.lower())
            self.state[x : x + len(word), y] |= direction.value

        if self.flipped:
            self.flip()

        return is_valid

    def get_letters(self, position: tuple, direction: Direction, length: int):
        do_unflip = False
        if direction == Direction.ACROSS and not self.flipped:
            self.flip()
            position = (position[1], position[0])
            do_unflip = True

        y, x = position
        letters = []

        for i, letter in enumerate(self.puzzle[x : x + length, y]):
            if letter == EMPTY_CELL:
                continue
            letters.append((i, letter))

        if do_unflip:
            self.flip()

        return letters

    def get_letter(self, position: tuple):
        return self.puzzle[position[1], position[0]]
