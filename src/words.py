import sqlite3
from typing import List, Tuple

import pandas as pd
from singleton_decorator import singleton

from word_grid import Direction


class Word(str):
    """Represents a crossword word placement"""

    def __new__(cls, word: pd.Series, position: Tuple[int, int], direction: Direction):
        """
        Args:
            word (pd.Series): Word data from the Word Index
            position (Tuple[int, int]): Where the word was placed in the grid
            direction (Direction): The direction in which the word was placed

        Returns:
            Word: A Word object
        """
        obj = str.__new__(cls, word.word)
        obj.meta = word
        obj.direction = direction
        obj.position = position
        return obj


@singleton
class WordIndex:
    def __init__(self) -> None:
        self.conn = sqlite3.connect("data/words.db")
        self.index = pd.read_sql("""
            SELECT 
                w.*,
                COUNT(d.id) AS num_definitions,
                COUNT(s.synonym_id) AS num_synonyms,
                COUNT(t.word_to_id) AS num_translations
            FROM 
                words w
                LEFT JOIN definitions d ON w.id = d.word_id
                LEFT JOIN synonyms s ON w.id = s.word_id
                LEFT JOIN translations t ON w.id = t.word_from_id
            GROUP BY 
                w.id                         
        """, self.conn)

    def get_translation(self, word: Word, lang_to: str) -> List[Word]:
        translations = pd.read_sql(
            f"""
            SELECT w2.*
            FROM translations t
            JOIN words w1 ON t.word_from_id = w1.id
            JOIN words w2 ON t.word_to_id = w2.id
            WHERE w1.id = '{word.meta.id}'; 
        """,
            self.conn,
        )

        if lang_to not in translations.language_code:
            return None

        words = []
        for _, row in translations[translations.language_code == lang_to].iterrows():
            words.append(Word(row, word.position, word.direction))

        return words

    def get_definition(self, word: Word) -> List[str]:
        definitions = pd.read_sql(
            f"""
            SELECT d.definition 
            FROM words w 
            JOIN definitions d ON w.id = d.word_id 
            WHERE w.id = {word.meta.id};
        """,
            self.conn,
        )

        return definitions.definition.to_list()

    def get_synonym(self, word: Word) -> List[Word]:
        synonyms = pd.read_sql(
            f"""
            SELECT w1.word AS original_word,
            w2.word AS synonym_word
            FROM 
            synonyms s
            JOIN words w1 ON s.word_id = w1.id
            JOIN words w2 ON s.synonym_id = w2.id
            WHERE 
            w1.id = {word.meta.id}; 
        """,
            self.conn,
        )

        words = []
        for _, row in synonyms.iterrows():
            words.append(Word(row, word.position, word.direction))

        return words

    def get_word(self, word: str, lang_code: str):
        return self.index[self.index.word == word & self.index.lang_code == lang_code]

    def get_data(self) -> pd.DataFrame:
        return self.index
