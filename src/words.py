import glob
import json
import linecache
import os
from typing import Tuple

import pandas as pd
from singleton_decorator import singleton
from tqdm import tqdm

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
        obj = str.__new__(cls, word.word.item())
        obj.meta = word
        obj.direction = direction
        obj.position = position
        return obj


@singleton
class WordIndex:
    def __init__(self) -> None:
        if os.path.exists("data/word_index.csv"):
            self.index = pd.read_csv("data/word_index.csv", encoding="utf-8")
        else:
            self.index = None

        if os.path.exists("data/word_freq.json"):
            with open("data/word_freq.json", "r", encoding="utf-8") as file:
                self.freq_data = json.load(file)
        else:
            self.freq_data = None
            
    def get_word_translation(self, word: Word, lang_to: str) -> Word:
        if lang_to not in word.meta.trans.fillna(""):
            return None
        
        word_translations = self.get_word_data(word)["translations"]
        for trans in word_translations:
            if trans["lang"] == lang_to:
                return self.get_word(trans["word"], lang_to)
            
    def get_word(self, word: str, lang_code: str):
        return self.index[self.index.word == word & self.index.lang_code == lang_code]
            
    def get_word_data(self, word: Word) -> dict:
        return json.loads(linecache.getline("data/raw-wiktextract-data.json", word.meta.index))

    def get_data(self) -> pd.DataFrame:
        if self.index is None:
            self.create_data()

        return self.index

    def extract_frequencies(self) -> None:
        files = glob.glob("./data/*/*-words.json")
        self.freq_data = {}
        for filename in files:
            with open(filename, "r", encoding="utf-8") as file:
                lang_code = os.path.basename(filename)[:2]
                if lang_code == "sp":
                    lang_code = "es"
                self.freq_data[lang_code] = json.load(file)

        with open("data/word_freq.json", "w", encoding="utf-8") as file:
            json.dump(self.freq_data, file)

    def create_data(self) -> None:
        if self.freq_data is None:
            self.extract_frequencies()

        columns = ["word", "index", "len", "lang_code", "pos", "freq", "cats", "trans"]
        word_index = []
        with open("./data/raw-wiktextract-data.json", "r", encoding="utf-8") as file:
            for index, line in tqdm(enumerate(file), total=9645555):
                data = json.loads(line)
                if "word" in data.keys():
                    if data["lang_code"] not in self.freq_data:
                        continue

                    word_index.append(
                        [
                            data["word"],
                            index,
                            len(data["word"]),
                            data["lang_code"],
                            data["pos"],
                            (
                                self.freq_data[data["lang_code"]][data["word"]]
                                if data["word"] in self.freq_data[data["lang_code"]]
                                else None
                            ),
                            data["categories"] if "categories" in data else None,
                            (
                                [t["code"] for t in data["translations"] if "code" in t]
                                if "translations" in data
                                else None
                            ),
                        ]
                    )

        self.index = pd.DataFrame(word_index, columns=columns)
        self.index.to_csv("word_index.csv", index=False)

