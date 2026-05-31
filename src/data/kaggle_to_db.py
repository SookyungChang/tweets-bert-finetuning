import pandas as pd
import sqlite3
from src.config import PathConfig

paths = PathConfig()

csv_file = paths.TWEETS_PATH / "training.1600000.processed.noemoticon.csv"
db_file = paths.DB_PATH / "kaggle_raw_data.db"

conn = sqlite3.connect(db_file)

columns = ['target', 'ids', 'date', 'flag', 'user', 'text']

chunk_size = 100000 

print("csv to db migration...")

# TextFileReader
for i, chunk in enumerate(pd.read_csv(csv_file, names=columns, header=None, encoding='latin-1', chunksize=chunk_size)):
    if i == 0:
        chunk.to_sql('kaggle_raw', conn, if_exists='replace', index=False) # first chunk
    else:
        chunk.to_sql('kaggle_raw', conn, if_exists='append', index=False)
        
    print(f"[{i+1:02d}] {(i+1)*chunk_size:,}th column db saved...")

conn.close()
print("Completed!")
