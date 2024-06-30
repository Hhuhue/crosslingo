import glob
import json
import os

import pandas as pd
from tqdm import tqdm
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from invoke import task

@task
def extract_word_frequencies(ctx):
    files = glob.glob("./data/*/*-words.txt")
    for filename in files:
        data = {}
        with open(filename, "r", encoding="utf-8") as file:
            for line in file:
                columns = [s.strip() for s in line.split("\t")]
                data[columns[1]] = int(columns[2])
                
        with open(filename.replace(".txt", ".json"), "w") as file:
            json.dump(data, file, indent=2)

@task
def create_word_index(ctx):
    files = glob.glob("./data/*/*-words.json")
    freq_data = {}
    for filename in files:
        with open(filename, "r", encoding="utf-8") as file:
            lang_code = os.path.basename(filename)[:2]
            if lang_code == "sp":
                lang_code = "es"
            freq_data[lang_code] = json.load(file)

    columns = ["word", "index", "len", "lang_code", "pos", "freq", "cats", "trans"]
    word_index = []
    with open("./data/raw-wiktextract-data.json", "r", encoding='utf-8') as file:
        for index, line in tqdm(enumerate(file), total=9645555):
            data = json.loads(line)
            if 'word' in data.keys():
                if data["lang_code"] not in freq_data:
                    continue
                
                word_index.append([
                    data['word'], 
                    index, 
                    len(data["word"]),
                    data["lang_code"], 
                    data["pos"], 
                    freq_data[data["lang_code"]][data["word"]] if data["word"] in freq_data[data["lang_code"]] else None,
                    data["categories"] if "categories" in data else None,
                    [t["code"] for t in data["translations"] if "code" in t] if "translations" in data else None
                ])
    pd.DataFrame(word_index, columns=columns).to_csv("word_index.csv")
    
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
        model.save(os.path.join(os.path.dirname(filename), 'word2vec.model'))
        
@task 
def test_word2vec(ctx):
    model = Word2Vec.load('data\deu_wikipedia_2016_1M\word2vec.model')

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
    similarity = model.wv.similarity(word, 'queen')
    print(f"Similarity between 'king' and 'queen': {similarity}")
