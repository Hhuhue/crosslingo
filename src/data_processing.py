import glob
import json
import os
import sqlite3

from tqdm import tqdm
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from invoke import task

def extract_frequencies() -> None:
    files = glob.glob("./data/*/*-words.json")
    freq_data = {}
    for filename in files:
        with open(filename, "r", encoding="utf-8") as file:
            lang_code = os.path.basename(filename)[:2]
            if lang_code == "sp":
                lang_code = "es"
            freq_data[lang_code] = json.load(file)

    with open("data/word_freq.json", "w", encoding="utf-8") as file:
        json.dump(freq_data, file)


def load_wiktextract():
    with open("data/word_freq.json", "r", encoding="utf-8") as file:
        freq_data = json.load(file)

    conn = sqlite3.connect("data/words.db")
    cursor = conn.cursor()

    categories_ref = {}

    for extract in glob.glob("data/*.jsonl"):
        lang_code = os.path.basename(extract).split("-")[0]

        count = 0
        with open(extract, "r", encoding="utf-8") as fp:
            for count, _ in enumerate(fp, 1):
                pass

        if count == 0:
            continue

        cursor.execute(
            """
            INSERT INTO sources (name, url)
            VALUES (?, ?)
            """,
            (extract, "https://kaikki.org/dictionary/rawdata.html"),
        )

        source_id = cursor.lastrowid

        with open(extract, "r", encoding="utf-8") as file:
            for index, line in tqdm(enumerate(file), total=count, desc=lang_code):
                data = json.loads(line)
                if "word" not in data.keys():
                    continue

                if data["lang_code"] != lang_code:
                    continue

                if (
                    lang_code in freq_data
                    and data["word"] in freq_data[data["lang_code"]]
                ):
                    freq = freq_data[lang_code][data["word"]]
                else:
                    freq = None

                word_data = (
                    data["word"],
                    index,
                    len(data["word"]),
                    data["lang_code"],
                    data["pos"],
                    freq,
                )

                cursor.execute(
                    """
                    INSERT INTO words (word, source_index, length, language_code, position, frequency)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    word_data,
                )

                word_id = cursor.lastrowid

                for sense in data["senses"]:
                    if "tags" in sense and (
                        "obsolete" in sense["tags"] or "no-gloss" in sense["tags"]
                    ):
                        continue

                    if "glosses" not in sense:
                        continue

                    cursor.execute(
                        """
                        INSERT INTO definitions (word_id, source_id, definition)
                        VALUES (?, ?, ?)
                        """,
                        (word_id, source_id, sense["glosses"][0]),
                    )

                if "categories" in data:
                    for category in data["categories"]:
                        if category not in categories_ref:
                            cursor.execute(
                                """
                                INSERT INTO categories (name)
                                VALUES (?)
                                """,
                                (category,),
                            )
                            categories_ref[category] = cursor.lastrowid

                        cat_data = (word_id, categories_ref[category])
                        cursor.execute(
                            "INSERT INTO word_categories VALUES (?, ?)", cat_data
                        )

        conn.commit()

    conn.close()


def load_wiktionary_traductions():
    conn = sqlite3.connect("data/words.db")
    cursor = conn.cursor()

    cursor.execute("SELECT id, language_code, source_index, position, word FROM words")

    rows = cursor.fetchall()
    word_lang_id = {}
    index_id = {}

    for word_id, lang_code, source_index, position, word in rows:
        if (word, position, lang_code) not in word_lang_id:
            word_lang_id[(word, position, lang_code)] = []

        if lang_code not in index_id:
            index_id[lang_code] = {}
        word_lang_id[(word, position, lang_code)].append(word_id)
        index_id[lang_code][source_index] = word_id

    for extract in glob.glob("data/*.jsonl"):
        lang_code = os.path.basename(extract).split("-")[0]

        count = 0
        with open(extract, "r", encoding="utf-8") as fp:
            for count, _ in enumerate(fp, 1):
                pass

        if count == 0:
            continue

        with open(extract, "r", encoding="utf-8") as file:
            for index, line in tqdm(enumerate(file), total=count, desc=lang_code):
                data = json.loads(line)
                if "word" not in data.keys():
                    continue

                if data["lang_code"] != lang_code:
                    continue

                if index not in index_id[lang_code]:
                    continue

                if "translations" not in data:
                    continue

                for translation in data["translations"]:
                    if "word" not in translation:
                        continue

                    trans_word = translation["word"]

                    if "code" in translation:
                        trans_code = translation["code"]
                    elif "lang_code" in translation:
                        trans_code = translation["lang_code"]
                    else:
                        continue

                    if trans_code not in index_id:
                        continue

                    if (trans_word, data["pos"], trans_code) not in word_lang_id:
                        continue

                    trans_ids = word_lang_id.get((trans_word, data["pos"], trans_code), [])
                    for trans_id in trans_ids:
                        try:
                            cursor.execute(
                                """
                                INSERT INTO translations VALUES (?, ?)
                                """,
                                (index_id[lang_code][index], trans_id)
                            )
                        except Exception as e:
                            print(f"Error inserting translation: {e}")
        conn.commit()
    conn.close()
    

def load_wiktionary_synonyms():
    conn = sqlite3.connect("data/words.db")
    cursor = conn.cursor()

    cursor.execute("SELECT id, language_code, source_index, position, word FROM words")

    rows = cursor.fetchall()
    word_lang_id = {}
    index_id = {}

    for word_id, lang_code, source_index, position, word in rows:
        if (word, position, lang_code) not in word_lang_id:
            word_lang_id[(word, position, lang_code)] = []
            
        if lang_code not in index_id:
            index_id[lang_code] = {}
        word_lang_id[(word, position, lang_code)].append(word_id)
        index_id[lang_code][source_index] = word_id

    for extract in glob.glob("data/*.jsonl"):
        lang_code = os.path.basename(extract).split("-")[0]

        count = 0
        with open(extract, "r", encoding="utf-8") as fp:
            for count, _ in enumerate(fp, 1):
                pass

        if count == 0:
            continue

        with open(extract, "r", encoding="utf-8") as file:
            for index, line in tqdm(enumerate(file), total=count, desc=lang_code):
                data = json.loads(line)
                if "word" not in data.keys():
                    continue

                if data["lang_code"] != lang_code:
                    continue

                if index not in index_id[lang_code]:
                    continue

                if "synonyms" not in data:
                    continue

                for synonym in data["synonyms"]:
                    if "word" not in synonym:
                        continue
 
                    syn_word = synonym["word"]

                    if (syn_word, data["pos"], lang_code) not in word_lang_id:
                        continue

                    syn_ids = word_lang_id.get((syn_word, data["pos"], lang_code), [])
                    for syn_id in syn_ids:
                        try:
                            cursor.execute(
                                """
                                INSERT INTO synonyms VALUES (?, ?)
                                """,
                                (index_id[lang_code][index], syn_id)
                            )
                        except Exception as e:
                            print(f"Error inserting translation: {e}")
        conn.commit()
    conn.close()


@task
def extract_word_frequencies(ctx):
    extract_frequencies()


@task
def create_word_index(ctx):
    load_wiktextract()
    load_wiktionary_synonyms()
    load_wiktionary_traductions()
    
@task
def train_word2vec(ctx):
    files = glob.glob("./data/*/*-sentences.txt")
    for filename in files:
        data = []
        with open(filename, "r", encoding="utf-8") as file:
            for line in file:
                data.append(line.split("\t")[1].strip())

        lang_code = os.path.basename(filename)[:2]

        if lang_code == "sp":
            language = "spanish"
        elif lang_code == "fr":
            language = "french"
        elif lang_code == "en":
            language = "english"
        elif lang_code == "de":
            language = "german"

        # Tokenize the text data
        stop_words = set(stopwords.words(language))
        sentences = []
        for sentence in tqdm(data):
            tokens = simple_preprocess(sentence.lower())
            tokens = [word for word in tokens if word not in stop_words]
            sentences.append(tokens)

        # Train the Word2Vec model
        model = Word2Vec(sentences, vector_size=200, window=7, min_count=1, workers=4)
        model.save(os.path.join(os.path.dirname(filename), "word2vec.model"))


@task
def test_word2vec(ctx):
    model = Word2Vec.load("data\deu_wikipedia_2016_1M\word2vec.model")

    # Example 1: Finding similar words
    word = "king"
    similar_words = model.wv.most_similar(word, topn=5)
    print(f"Words similar to '{word}': {similar_words}")

    # Example 2: Finding analogies
    word_p = "woman"
    word_n = "man"
    analogy = model.wv.most_similar(positive=[word_p, word], negative=[word_n], topn=1)
    print(f"Analogy '{word_p}' is to '{word_n}' as '{word}' is to: {analogy}")

    # Example 3: Checking word similarity
    similarity = model.wv.similarity(word, "queen")
    print(f"Similarity between 'king' and 'queen': {similarity}")
