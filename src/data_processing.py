import glob
import os

from tqdm import tqdm
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from invoke import task

from words import WordIndex

@task
def extract_word_frequencies(ctx):
    index = WordIndex()
    index.extract_frequencies()

@task
def create_word_index(ctx):
    index = WordIndex()
    index.create_data()

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
