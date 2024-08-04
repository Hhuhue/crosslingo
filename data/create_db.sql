DROP TABLE IF EXISTS words;

CREATE TABLE words (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_index INTEGER NOT NULL,
    word TEXT NOT NULL,
    length INTEGER NOT NULL,
    language_code TEXT NOT NULL,
    position TEXT NOT NULL,
    frequency REAL NULL
);

DROP TABLE IF EXISTS sources;

CREATE TABLE sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    url TEXT NOT NULL
);

DROP TABLE IF EXISTS definitions;

CREATE TABLE definitions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    word_id INTEGER NOT NULL,
    source_id INTEGER NOT NULL,
    definition TEXT NOT NULL,
    FOREIGN KEY (word_id) REFERENCES words (id)
    FOREIGN KEY (source_id) REFERENCES sources (id)
);

DROP TABLE IF EXISTS categories;

CREATE TABLE categories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL
);

DROP TABLE IF EXISTS word_categories;

CREATE TABLE word_categories (
    word_id INTEGER NOT NULL,
    category_id INTEGER NOT NULL,
    PRIMARY KEY (word_id, category_id),
    FOREIGN KEY (word_id) REFERENCES words (id),
    FOREIGN KEY (category_id) REFERENCES word_categories (id)
);

DROP TABLE IF EXISTS translations;

CREATE TABLE translations (
    word_from_id INTEGER NOT NULL,
    word_to_id INTEGER NOT NULL,
    PRIMARY KEY (word_from_id, word_to_id),
    FOREIGN KEY (word_from_id) REFERENCES words (id),
    FOREIGN KEY (word_to_id) REFERENCES words (id)
);

DROP TABLE IF EXISTS synonyms;

CREATE TABLE synonyms (
    word_id INTEGER NOT NULL,
    synonym_id INTEGER NOT NULL,
    PRIMARY KEY (word_id, synonym_id),
    FOREIGN KEY (word_id) REFERENCES words (id),
    FOREIGN KEY (synonym_id) REFERENCES words (id)
);