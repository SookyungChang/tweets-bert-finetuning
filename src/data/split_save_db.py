import sqlite3
import pandas as pd
from src.config import PathConfig

paths = PathConfig()

conn = sqlite3.connect(paths.DB_PATH / "kaggle_raw_data.db")

# ==========================================
# 1. TRAIN 320k (train.db)
# ==========================================
print("Train data...")
train_query = """
WITH Ranked AS (
    SELECT ids, text, target as label,
           ROW_NUMBER() OVER (PARTITION BY target ORDER BY RANDOM()) as rn
    FROM clean_tweets
)
SELECT ids, text, label FROM Ranked WHERE rn <= 160000; 
"""

df_train = pd.read_sql(train_query, conn)
conn_train = sqlite3.connect(paths.DB_PATH / "train.db")
df_train.to_sql('tweets', conn_train, if_exists='replace', index=False)
conn_train.close()

# ==========================================
# 2. VAL 160k (val.db)
# ==========================================
print("Val data...")
val_query = f"""
WITH Ranked AS (
    SELECT ids, text, target as label,
           ROW_NUMBER() OVER (PARTITION BY target ORDER BY RANDOM()) as rn
    FROM clean_tweets
    WHERE ids NOT IN ({','.join(map(str, df_train['ids']))})
)
SELECT ids, text, label FROM Ranked WHERE rn <= 80000; 
"""
df_val = pd.read_sql(val_query, conn)
conn_val = sqlite3.connect(paths.DB_PATH / "val.db")
df_val.to_sql('tweets', conn_val, if_exists='replace', index=False)
conn_val.close()

# ==========================================
# 3. TEST 160k (test.db)
# ==========================================
print("Test data...")
used_ids = set(df_train['ids']).union(set(df_val['ids']))
test_query = f"""
WITH Ranked AS (
    SELECT ids, text, target as label,
           ROW_NUMBER() OVER (PARTITION BY target ORDER BY RANDOM()) as rn
    FROM clean_tweets
    -- 💡 Train과 Val에 이미 쓰인 ID는 전부 제외!
    WHERE ids NOT IN ({','.join(map(str, used_ids))})
)
SELECT ids, text, label FROM Ranked WHERE rn <= 80000; 
"""
df_test = pd.read_sql(test_query, conn)
conn_test = sqlite3.connect(paths.DB_PATH / "test.db")
df_test.to_sql('tweets', conn_test, if_exists='replace', index=False)
conn_test.close()

conn.close()
print("Split Completed.")
