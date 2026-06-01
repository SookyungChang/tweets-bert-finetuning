import pandas as pd
import sqlite3
import re
from langdetect import detect_langs
from langdetect.lang_detect_exception import LangDetectException
from src.config import PathConfig

paths = PathConfig()

# Preprocess
def remove_urls(text):
    url_pattern = r"https?://\S+|www\.\S+"
    return re.sub(url_pattern, "", str(text)).strip()

def safe_detect(text):
    if not text or len(str(text)) <= 5:
        return False
    try:
        res = detect_langs(text)[0]
        if res.lang == "en" and res.prob > 0.9:
            return True
    except LangDetectException:
        pass
    return False

def clean_database(db_path):
    conn = sqlite3.connect(db_path)
    
    chunk_size = 100000
    
    query = "SELECT ids, text, target FROM kaggle_raw"
    
    for i, chunk in enumerate(pd.read_sql(query, conn, chunksize=chunk_size)):
        # clean lable
        chunk["target"] = chunk["target"].replace(4, 1)
        
        # remove URLs
        chunk["text"] = chunk["text"].apply(remove_urls)
        
        # filter only english comments
        is_english = chunk["text"].apply(safe_detect)
        chunk = chunk[is_english].reset_index(drop=True)

        if i == 0:
            chunk.to_sql('clean_tweets', conn, if_exists='replace', index=False)
        else:
            chunk.to_sql('clean_tweets', conn, if_exists='append', index=False)
            
        print(f"[{i+1:02d}] {(i+1)*chunk_size:,} saved...")
        
    conn.close()
    print("Completed!")

import sqlite3
import pandas as pd
import re

def remove_mentions(text):
    mention_pattern = r"@\S+"
    return re.sub(mention_pattern, "", str(text)).strip()

def clean_mentions_in_separated_dbs(db_path):
    conn = sqlite3.connect(db_path)

    chunk_size = 100000

    print("remove commenting @...")

    query = "SELECT ids, text, target FROM clean_tweets"

    for i, chunk in enumerate(pd.read_sql(query, conn, chunksize=chunk_size)):

        chunk["text"] = chunk["text"].apply(remove_mentions)
        chunk = chunk[chunk["text"].str.len() > 5].reset_index(drop=True)

        if i == 0:
            chunk.to_sql('no_com_tweets', conn, if_exists='replace', index=False)
        else:
            chunk.to_sql('no_com_tweets', conn, if_exists='append', index=False)
            
        print(f"[{i+1:02d}] {(i+1)*chunk_size:,} saved...")
        
    conn.close()
    print("Completed!")
    

if __name__ == "__main__":
    db_path=paths.DB_PATH / "kaggle_raw_data.db"
    # clean_database(db_path)
    clean_mentions_in_separated_dbs(db_path)
