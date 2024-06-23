import unittest
import logging
import pytest
from word_grid import WordGrid, Direction, EMPTY_CELL

TEST_LOG = "tests/data/test.log"

class TestWordGrid(unittest.TestCase):
    
    def get_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        console_handler = logging.FileHandler(TEST_LOG, mode="w")
        console_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        return logger

    def test_init_should_create_letter_and_state_arrays(self):
        # Arrange
        shape = (5, 10)
        
        # Action
        grid = WordGrid(shape)
        
        # Assert
        self.assertEqual(tuple(grid.shape), shape)
        self.assertEqual(grid.puzzle.shape, shape)
        self.assertEqual(grid.state.shape, shape)
        self.assertTrue((grid.puzzle == EMPTY_CELL).all())
        self.assertTrue((grid.state == Direction.NONE.value).all())

    def test_flip_should_transpose_arrays_and_shape(self):
        # Arrange
        shape = (5, 10)
        flipped_shape = (shape[1], shape[0])
        grid = WordGrid(shape)
        
        # Action
        grid.flip()
        
        # Assert
        self.assertEqual(tuple(grid.shape), flipped_shape)
        self.assertEqual(grid.puzzle.shape, flipped_shape)
        self.assertEqual(grid.state.shape, flipped_shape)
        self.assertTrue(grid.flipped)
        
    def test_reset_should_erase_arrays_content(self):
        # Arrange
        shape = (5, 10)
        word = "hat"
        logger = self.get_logger()
        grid = WordGrid(shape, logger)
        grid.puzzle[1:1 + len(word), 1] = list(word.lower())
        grid.state[1:1 + len(word), 1] |= Direction.DOWN.value
        
        # Action
        grid.reset()
        
        # Assert
        self.assertTrue((grid.puzzle == EMPTY_CELL).all())
        self.assertTrue((grid.state == Direction.NONE.value).all())

    def test_validate_word_should_return_false_when_word_out_of_bound(self):
        # Arrange
        shape = (5, 10)
        position = (7, 2)
        word = "test"
        logger = self.get_logger()
        grid = WordGrid(shape, logger)
        
        # Action
        across_valid = grid.validate_word(position, Direction.ACROSS, word)
        down_valid = grid.validate_word(position, Direction.DOWN, word)
        
        # Assert
        self.assertFalse(across_valid)
        self.assertFalse(down_valid)
        with open(TEST_LOG, "r") as file:
            lines = file.readlines()
        self.assertEqual(len(lines), 2)
        self.assertTrue(all(["length" in line for line in lines]))
        
    def test_validate_word_should_return_false_when_a_word_is_overlapped(self):
        # Arrange
        shape = (5, 10)
        word = "hat"
        logger = self.get_logger()
        grid = WordGrid(shape, logger)
        grid.puzzle[1:1 + len(word), 1] = list(word.lower())
        grid.state[1:1 + len(word), 1] |= Direction.DOWN.value
        grid.puzzle[2, 4:4 + len(word)] = list(word.lower())
        grid.state[2, 4:4 + len(word)] |= Direction.ACROSS.value
        overlapping_word = "chats"
        
        # Action
        across_valid = grid.validate_word((3, 2), Direction.ACROSS, overlapping_word)
        down_valid = grid.validate_word((1, 0), Direction.DOWN, overlapping_word)
        
        # Assert
        self.assertFalse(across_valid)
        self.assertFalse(down_valid)
        with open(TEST_LOG, "r") as file:
            lines = file.readlines()
        self.assertEqual(len(lines), 2)
        self.assertTrue(all(["overlap" in line for line in lines]))
        
    def test_validate_word_should_return_false_when_a_word_is_at_start_of_end_of_word_to_validate(self):
        # Arrange
        shape = (5, 10)
        word = "hello"
        logger = self.get_logger()
        grid = WordGrid(shape, logger)

        grid.puzzle[2, 2:2 + len(word)] = list(word.lower())
        grid.state[2, 2:2 + len(word)] |= Direction.ACROSS.value
        interfering_word = "on"
        
        # Action
        word_before_across_valid = grid.validate_word((0, 2), Direction.ACROSS, interfering_word)
        word_after_across_valid = grid.validate_word((7, 2), Direction.ACROSS, interfering_word)
        word_before_down_valid = grid.validate_word((4, 0), Direction.DOWN, interfering_word)
        word_after_down_valid = grid.validate_word((3, 3), Direction.DOWN, interfering_word)
        
        # Assert
        self.assertFalse(word_before_across_valid)
        self.assertFalse(word_after_across_valid)
        self.assertFalse(word_before_down_valid)
        self.assertFalse(word_after_down_valid)
        with open(TEST_LOG, "r") as file:
            lines = file.readlines()
        self.assertEqual(len(lines), 4)
        self.assertTrue(all(["interference" in line for line in lines]))
        
    def test_validate_word_should_return_true_when_word_in_bounds_in_empty_grid(self):
        # Arrange
        shape = (10, 10)
        word = "gated"
        grid = WordGrid(shape)
        
        # Action
        is_valid_across = grid.validate_word((3, 2), Direction.ACROSS, word)
        is_valid_down = grid.validate_word((0, 6), Direction.DOWN, word)
        
        # Assert
        self.assertTrue(is_valid_across)
        self.assertTrue(is_valid_down)

    def test_add_word(self):
        # Test adding a word to the grid
        self.assertTrue(self.grid.add_word((0, 0), Direction.ACROSS, "hello"))
        self.assertEqual(self.grid.puzzle[0, 0:5], list("hello"))
        self.assertEqual(self.grid.state[0, 0:5], [Direction.ACROSS.value] * 5)

    def test_get_letters(self):
        # Test getting letters from the grid
        self.grid.add_word((0, 0), Direction.ACROSS, "hello")
        self.assertEqual(self.grid.get_letters((0, 0), Direction.ACROSS, 5), [(0, 'h'), (1, 'e'), (2, 'l'), (3, 'l'), (4, 'o')])

    def test_get_letter(self):
        # Test getting a single letter from the grid
        self.grid.add_word((0, 0), Direction.ACROSS, "hello")
        self.assertEqual(self.grid.get_letter((0, 0)), 'h')

if __name__ == '__main__':
    unittest.main()