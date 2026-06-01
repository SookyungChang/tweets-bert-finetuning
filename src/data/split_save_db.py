import sqlite3
import pandas as pd
from src.config import PathConfig

paths = PathConfig()

conn = sqlite3.connect(paths.DB_PATH / "kaggle_raw_data.db")
conn_new = sqlite3.connect(paths.DB_PATH / "tweets.db")

table_name = 'no_com_tweets'

# ==========================================
# 1. TRAIN 320k (train.db)
# ==========================================
print("Train data...")
train_query = f"""
WITH Ranked AS (
    SELECT ids, text, target as label,
           ROW_NUMBER() OVER (PARTITION BY target ORDER BY RANDOM()) as rn
    FROM {table_name}
)
SELECT ids, text, label FROM Ranked WHERE rn <= 160000; 
"""

df_train = pd.read_sql(train_query, conn)
df_train.to_sql('train', conn_new, if_exists='replace', index=False)

# ==========================================
# 2. VAL 160k (val.db)
# ==========================================
print("Val data...")
val_query = f"""
WITH Ranked AS (
    SELECT ids, text, target as label,
           ROW_NUMBER() OVER (PARTITION BY target ORDER BY RANDOM()) as rn
    FROM {table_name}
    WHERE ids NOT IN ({','.join(map(str, df_train['ids']))})
)
SELECT ids, text, label FROM Ranked WHERE rn <= 80000; 
"""
df_val = pd.read_sql(val_query, conn)
df_val.to_sql('val', conn_new, if_exists='replace', index=False)

# ==========================================
# 3. TEST 160k (test.db)
# ==========================================
print("Test data...")
used_ids = set(df_train['ids']).union(set(df_val['ids']))
test_query = f"""
WITH Ranked AS (
    SELECT ids, text, target as label,
           ROW_NUMBER() OVER (PARTITION BY target ORDER BY RANDOM()) as rn
    FROM {table_name}
    WHERE ids NOT IN ({','.join(map(str, used_ids))})
)
SELECT ids, text, label FROM Ranked WHERE rn <= 80000; 
"""
df_test = pd.read_sql(test_query, conn)
df_test.to_sql('test', conn_new, if_exists='replace', index=False)

conn_new.close()
conn.close()
print("Split Completed.")
