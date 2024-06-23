import glob
import json
import os

import pandas as pd
from tqdm import tqdm

def extract_word_frequencies():
    files = glob.glob("./data/*/*-words.txt")
    for filename in files:
        data = {}
        with open(filename, "r", encoding="utf-8") as file:
            for line in file:
                columns = [s.strip() for s in line.split("\t")]
                data[columns[1]] = int(columns[2])
                
        with open(filename.replace(".txt", ".json"), "w") as file:
            json.dump(data, file, indent=2)
            
def create_word_index():
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
    


if __name__ == "__main__":
    extract_word_frequencies()
    create_word_index()