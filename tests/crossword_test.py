from io import StringIO
import unittest
from unittest.mock import MagicMock, patch

from loguru import logger
import pytest
import pandas as pd

from crossword import Crossword, CluesMode, CrosswordGenerator, CrosswordStyle
from words import Word, Direction

class CrosswordTest(unittest.TestCase):
    def setUp(self):
        self.stream = StringIO()
        logger.add(self.stream, level="DEBUG")

        columns = [
            "id",
            "source_index",
            "word",
            "length",
            "language_code",
            "position",
            "frequency",
            "num_en",
            "num_de",
            "num_fr",
            "num_es",
            "num_definitions",
            "num_synonyms",
        ]
        test_words = [
            [0, 100, "Katze", 5, "de", "noun", 145.0, 1, 0, 1, 1, 1, 1],
            [1, 100, "cat",   3, "en", "noun", 309.0, 0, 1, 1, 1, 1, 1],
            [2, 200, "gato",  4, "es", "noun", 353.0, 1, 1, 1, 0, 1, 1],
            [3, 200, "chat",  4, "fr", "noun", 349.0, 1, 1, 0, 1, 1, 1],
            [4, 300, "Hund",  4, "de", "noun", 145.0, 1, 0, 1, 1, 1, 1],
            [5, 300, "dog",   3, "en", "noun", 309.0, 0, 1, 1, 1, 1, 1],
            [6, 400, "perro", 5, "es", "noun", 353.0, 1, 1, 1, 0, 1, 1],
            [7, 400, "chien", 5, "fr", "noun", 349.0, 1, 1, 0, 1, 1, 1],
            [8, 500, "Test",  4, "de", "noun", 0.0, 0, 0, 0, 0, 0, 0],
            [9, 500, "test",  4, "en", "noun", 0.0, 0, 0, 0, 0, 0, 0],
            [10, 600, "test", 4, "es", "noun", 0.0, 0, 0, 0, 0, 0, 0],
            [11, 600, "test", 4, "fr", "noun", 0.0, 0, 0, 0, 0, 0, 0],
        ]
        self.test_index = pd.DataFrame(test_words, columns=columns)

        self.test_definitions = [
            "Familie der Feliden aus der Ordnung der Raubtiere mit mehreren.",
            "A small domesticated carnivorous mammal.",
            "Mamífero de la familia de los félidos.",
            "Petit mammifère carnivore domestique.",
            "Haustier, dessen Vorfahre der Wolf ist.",
            "A mammal of the family Canidae.",
            "Variedad doméstica del lobo de muchas y diversas razas, compañero del hombre desde tiempos prehistóricos."
            "Mammifère carnivore de la famille des Canidés, apparenté au loup."
        ]

        self.test_synonyms = ["Mieze", "kitty", "minino", "minet", "Köter", "mutt", "chucho", "cabot"]


class TestCrossword(CrosswordTest):


    @patch("crossword.WordIndex")
    def test_init_should_load_definitions_when_clues_mode_is_definition(
        self, mock_index: MagicMock
    ):
        # Arrange
        test_word = Word(self.test_index.iloc[0], (0, 0), Direction.ACROSS)
        test_definitions = [self.test_definitions[0]]
        mock_index.return_value.get_definition.return_value = test_definitions

        # Action
        crossword = Crossword(
            None, [test_word], test_word.meta.language_code, CluesMode.DEFINITION
        )

        # Assert
        self.assertEqual(test_definitions, crossword.clues)

    @patch("crossword.WordIndex")
    def test_init_should_load_synonyms_when_clues_mode_is_synonym(
        self, mock_index: MagicMock
    ):
        # Arrange
        test_word = Word(self.test_index.iloc[1], (0, 0), Direction.ACROSS)
        test_synonyms = [self.test_synonyms[1]]
        mock_index.return_value.get_synonym.return_value = test_synonyms

        # Action
        crossword = Crossword(
            None, [test_word], test_word.meta.language_code, CluesMode.SYNONYM
        )

        # Assert
        self.assertEqual(test_synonyms, crossword.clues)

    @patch("crossword.WordIndex")
    def test_init_should_load_translations_when_word_lang_code_and_lang_from_are_different(
        self, mock_index: MagicMock
    ):
        # Arrange
        test_word = Word(self.test_index.iloc[2], (0, 0), Direction.ACROSS)
        test_translation = Word(
            self.test_index.iloc[0], test_word.position, test_word.direction
        )
        mock_index.return_value.get_translation.return_value = [test_translation]

        # Action
        crossword = Crossword(None, [test_word], "de", CluesMode.TRANSLATION)

        # Assert
        self.assertEqual([test_translation], crossword.clues)


class TestCrosswordGenerator(CrosswordTest):

    def mock_get_definition(self, word: Word):
        if word.meta.id < len(self.test_definitions):
            return [self.test_definitions[word.meta.id]]

        return None

    def mock_get_synonym(self, word: Word):
        if word.meta.id < len(self.test_synonyms):
            return [self.test_synonyms[word.meta.id]]

        return None
    
    def mock_get_translation(self, word: Word, lang_to: str):
        lang_offset = {
            "de": 0,
            "en": 1,
            "es": 2,
            "fr": 3
        }
        
        offset = lang_offset[lang_to] - lang_offset[word.meta.language_code]
        return [Word(self.test_index.iloc[word.meta.id + offset], word.position, word.direction)]
        
    @patch("crossword.WordIndex")
    def test_generate_should_ignore_words_with_no_definitions_when_clues_mode_is_definition(self, mock_index: MagicMock):
        # Arrange
        mock_word_index = MagicMock()
        mock_word_index.get_definition = self.mock_get_definition
        mock_index.return_value = mock_word_index

        # Action
        generator = CrosswordGenerator(self.test_index, CrosswordStyle.BRITISH, 123)
        result = generator.generate((5,5), "en", 3)

        # Assert
        self.assertEqual(2, len(result.words))
        self.assertTrue(self.test_index.loc[1, 'word'] in result.words)
        self.assertTrue(self.test_index.loc[5, 'word'] in result.words)
        self.assertTrue(self.test_definitions[1] in result.clues)
        self.assertTrue(self.test_definitions[5] in result.clues)
        
    @patch("crossword.WordIndex")
    def test_generate_should_ignore_words_with_no_synonyms_when_clues_mode_is_synonym(self, mock_index: MagicMock):
        # Arrange
        mock_word_index = MagicMock()
        mock_word_index.get_synonym = self.mock_get_synonym
        mock_index.return_value = mock_word_index

        # Action
        generator = CrosswordGenerator(self.test_index, CrosswordStyle.BRITISH, 123)
        result = generator.generate((5,5), "fr", 2, clues_mode=CluesMode.SYNONYM)

        # Assert
        self.assertEqual(2, len(result.words))
        self.assertTrue(self.test_index.loc[3, 'word'] in result.words)
        self.assertTrue(self.test_index.loc[7, 'word'] in result.words)
        self.assertTrue(self.test_synonyms[3] in result.clues)
        self.assertTrue(self.test_synonyms[7] in result.clues)
        

    @patch("crossword.WordIndex")
    def test_generate_should_ignore_words_with_no_translations_when_clues_mode_is_translation(self, mock_index: MagicMock):
        # Arrange
        mock_word_index = MagicMock()
        mock_word_index.get_translation = self.mock_get_translation
        mock_index.return_value = mock_word_index

        # Action
        generator = CrosswordGenerator(self.test_index, CrosswordStyle.BRITISH, 1)
        result = generator.generate((5,5), "es", 2, "de", clues_mode=CluesMode.TRANSLATION)

        # Assert
        self.assertEqual(2, len(result.words))
        self.assertTrue(self.test_index.loc[0, 'word'] in result.words)
        self.assertTrue(self.test_index.loc[4, 'word'] in result.words)
        self.assertTrue(self.test_index.loc[2, 'word'] in result.clues)
        self.assertTrue(self.test_index.loc[6, 'word'] in result.clues)