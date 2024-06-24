import unittest
import logging
import pytest
from word_grid import WordGrid, Direction, EMPTY_CELL
from io import StringIO

class TestWordGrid(unittest.TestCase):
    
    def setUp(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        self.stream = StringIO()
        console_handler = logging.StreamHandler(self.stream)
        console_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        self.logger = logger

    def test_init_should_create_letter_and_state_arrays(self):
        # Arrange
        shape = (5, 10)
        
        # Action
        grid = WordGrid(shape, self.logger)
        
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
        grid = WordGrid(shape, self.logger)
        grid.puzzle[1:1 + len(word), 1] = list(word)
        grid.state[1:1 + len(word), 1] |= Direction.DOWN.value
        
        # Action
        grid.reset()
        
        # Assert
        self.assertTrue((grid.puzzle == EMPTY_CELL).all())
        self.assertTrue((grid.state == Direction.NONE.value).all())
        
    def test_get_letters_should_return_non_empty_letters_in_a_row_or_column_from_a_position(self):
        # Arrange
        shape = (5, 10)
        grid = WordGrid(shape, self.logger)
        word_1 = "eagle"
        word_2 = "angle"
        word_3 = "pet"
        grid.puzzle[0:0 + len(word_1), 1] = list(word_1)
        grid.state[0:0 + len(word_1), 1] |= Direction.DOWN.value
        grid.puzzle[0:0 + len(word_2), 3] = list(word_2)
        grid.state[0:0 + len(word_2), 3] |= Direction.DOWN.value
        grid.puzzle[1, 5:5 + len(word_3)] = list(word_3)
        grid.state[1, 5:5 + len(word_3)] |= Direction.ACROSS.value
        
        # Action
        letters_across = grid.get_letters((0, 1), Direction.ACROSS, 6)
        letters_down = grid.get_letters((7, 0), Direction.DOWN, 2)
        
        # Assert
        self.assertEqual(letters_across, [(1, word_1[1]), (3, word_2[1]), (5, word_3[0])])
        self.assertEqual(letters_down, [(1, word_3[2])])

    def test_validate_word_should_return_false_when_word_out_of_bound(self):
        # Arrange
        shape = (5, 10)
        position = (7, 2)
        word = "test"
        grid = WordGrid(shape, self.logger)
        
        # Action
        across_valid = grid.validate_word(position, Direction.ACROSS, word)
        down_valid = grid.validate_word(position, Direction.DOWN, word)
        
        # Assert
        self.assertFalse(across_valid)
        self.assertFalse(down_valid)
        
        self.stream.seek(0)
        lines = self.stream.readlines()
        self.assertEqual(len(lines), 2)
        self.assertTrue(all(["length" in line for line in lines]))
        
    def test_validate_word_should_return_false_when_a_word_is_overlapped(self):
        # Arrange
        shape = (5, 10)
        word = "hat"
        grid = WordGrid(shape, self.logger)
        grid.puzzle[1:1 + len(word), 1] = list(word)
        grid.state[1:1 + len(word), 1] |= Direction.DOWN.value
        grid.puzzle[2, 4:4 + len(word)] = list(word)
        grid.state[2, 4:4 + len(word)] |= Direction.ACROSS.value
        overlapping_word = "chats"
        
        # Action
        across_valid = grid.validate_word((3, 2), Direction.ACROSS, overlapping_word)
        down_valid = grid.validate_word((1, 0), Direction.DOWN, overlapping_word)
        
        # Assert
        self.assertFalse(across_valid)
        self.assertFalse(down_valid)
        
        self.stream.seek(0)
        lines = self.stream.readlines()
        self.assertEqual(len(lines), 2)
        self.assertTrue(all(["overlap" in line for line in lines]))
        
    def test_validate_word_should_return_false_when_a_word_is_at_start_or_end_of_word_to_validate(self):
        # Arrange
        shape = (5, 10)
        word = "hello"
        grid = WordGrid(shape, self.logger)

        grid.puzzle[2, 2:2 + len(word)] = list(word)
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
        
        self.stream.seek(0)
        lines = self.stream.readlines()
        self.assertEqual(len(lines), 4)
        self.assertTrue(all(["interference" in line for line in lines]))
        
    def test_validate_word_should_return_false_when_crossing_word_with_different_letters(self):
        # Arrange
        shape = (5, 10)
        word_1 = "hello"
        word_2 = "halo"
        grid = WordGrid(shape, self.logger)

        grid.puzzle[1, 2:2 + len(word_1)] = list(word_1)
        grid.state[1, 2:2 + len(word_1)] |= Direction.ACROSS.value
        grid.puzzle[1:1 + len(word_2), 2] = list(word_2)
        grid.state[1:1 + len(word_2), 2] |= Direction.DOWN.value
        crossing_word = "cart"
        
        # Action
        across_valid = grid.validate_word((0, 2), Direction.ACROSS, crossing_word)
        down_valid = grid.validate_word((3, 0), Direction.DOWN, crossing_word)
        
        # Assert
        self.assertFalse(across_valid)
        self.assertFalse(down_valid)
        
        self.stream.seek(0)
        lines = self.stream.readlines()
        print(lines)
        self.assertEqual(len(lines), 2)
        self.assertTrue(all(["conflict" in line for line in lines]))
        
    def test_validate_word_should_return_false_when_new_word_sides_touches_other_word(self):
        # Arrange
        shape = (5, 10)
        word_1 = "hello"
        word_2 = "halo"
        grid = WordGrid(shape, self.logger)

        grid.puzzle[1, 2:2 + len(word_1)] = list(word_1)
        grid.state[1, 2:2 + len(word_1)] |= Direction.ACROSS.value
        grid.puzzle[1:1 + len(word_2), 2] = list(word_2)
        grid.state[1:1 + len(word_2), 2] |= Direction.DOWN.value
        interfering_word = "cart"
        
        # Action
        across_valid = grid.validate_word((0, 0), Direction.ACROSS, interfering_word)
        down_valid = grid.validate_word((7, 0), Direction.DOWN, interfering_word)
        
        # Assert
        self.assertFalse(across_valid)
        self.assertFalse(down_valid)
        
        self.stream.seek(0)
        lines = self.stream.readlines()
        print(lines)
        self.assertEqual(len(lines), 2)
        self.assertTrue(all(["Side" in line for line in lines]))
        
    def test_validate_word_should_return_true_when_word_in_bounds_in_empty_grid(self):
        # Arrange
        shape = (5, 10)
        word = "gated"
        grid = WordGrid(shape, self.logger)
        
        # Action
        is_valid_across = grid.validate_word((3, 2), Direction.ACROSS, word)
        is_valid_down = grid.validate_word((6, 0), Direction.DOWN, word)
        
        # Assert
        self.assertTrue(is_valid_across)
        self.assertTrue(is_valid_down)
        
    def test_validate_word_should_return_true_when_word_crosses_other_word_without_conflict(self):
        # Arrange
        shape = (5, 10)
        word_1 = "great"
        word_2 = "recall"
        grid = WordGrid(shape, self.logger)
        grid.puzzle[0:len(word_1), 2] = list(word_1)
        grid.state[0:len(word_1), 2] |= Direction.DOWN.value
        grid.puzzle[1, 2:2 + len(word_2)] = list(word_2)
        grid.state[1, 2:2 + len(word_2)] |= Direction.ACROSS.value
        crossing_word = "cat"

        # Action
        is_valid_across = grid.validate_word((1, 3), Direction.ACROSS, crossing_word)
        is_valid_down = grid.validate_word((5, 0), Direction.DOWN, crossing_word)
        
        # Assert
        self.assertTrue(is_valid_across)
        self.assertTrue(is_valid_down)

if __name__ == '__main__':
    unittest.main()