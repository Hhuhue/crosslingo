# CrossLingo

A crossword puzzle generator aiming to makes guesses from words definition or their translation in other languages. The aim is to provide an innovative way to learn a language.

## Setup

**Python environment:**

```
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install poetry
poetry install
pip install -e .
```

**Download data:**

* Wiktionary words from [wikiextract ](https://github.com/tatuylonen/wiktextract)(2.2GB): https://kaikki.org/dictionary/raw-wiktextract-data.json
* Language corpus from [wortschatz](https://www.wortschatz.uni-leipzig.de/de/download) (~200MB each)):
  * French corpus: https://downloads.wortschatz-leipzig.de/corpora/fra_wikipedia_2021_1M.tar.gz
  * English corpus: https://downloads.wortschatz-leipzig.de/corpora/eng_wikipedia_2016_1M.tar.gz
  * Spanish corpus: https://downloads.wortschatz-leipzig.de/corpora/spa_wikipedia_2021_1M.tar.gz
  * German corpus: https://downloads.wortschatz-leipzig.de/corpora/deu_wikipedia_2021_1M.tar.gz
  * ...

Extract the files to `./data` then run the preprocessing script

```
python src/data_processing.py
```
