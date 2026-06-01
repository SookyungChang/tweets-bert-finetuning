import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
print(torch.cuda.is_available())
import pandas as pd
import sqlite3
from datetime import datetime
from huggingface_hub import snapshot_download

from src.data.preprocess import build_dataset
from sklearn.metrics import f1_score
from src.config_bert import ModelConfig
from src.config import PathConfig
from src.inference import predictor_base, predictor_bert

modelconf = ModelConfig()
paths = PathConfig()

def base(X, y):
    model_pipe = predictor_base.load_model()
    preds = model_pipe.predict(X)
    probs = model_pipe.predict_proba(X)
    f1 = f1_score(y, preds, average=modelconf.f1_avg)
    return preds, f1, probs

bert_path = snapshot_download(repo_id=modelconf.fine_tuned_model)
print("bert model is loaded")   

def bert(df):
    bert = predictor_bert.Predictor(bert_path)
    df = bert.predict_df(df)
    return df

def predict_to_db_by_batch(db_name=paths.DB_PATH / "explorations.db", chunk_size=10000):
    conn = sqlite3.connect(paths.DB_PATH / "tweets.db")
    # sample_size = 10000
    # test_df = pd.read_sql(f"SELECT * FROM test ORDER BY RANDOM() LIMIT {sample_size}", conn)
    test_df = pd.read_sql(f"SELECT * FROM test ORDER BY RANDOM()", conn)
    X_test, y_test = test_df['text'], test_df['label']
    ids = test_df['ids']
    conn.close()

    db_path = paths.DB_PATH.parent / db_name
    conn = sqlite3.connect(db_path)
    
    total_samples = len(X_test)
    print(f"total {total_samples} data by chunk size {chunk_size}...")
    
    for i in range(0, total_samples, chunk_size):
        start_idx = i
        end_idx = min(i + chunk_size, total_samples)

        chunk_X = X_test[start_idx:end_idx]
        chunk_y = y_test[start_idx:end_idx]
        chunk_ids = ids[start_idx:end_idx]
        
        preds_chunk, f1_base, probs = base(chunk_X, chunk_y)

        chunk_df = pd.DataFrame({
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'ids': chunk_ids,
            'text': chunk_X,
            'base_pred': preds_chunk,
            'label': chunk_y,
            'base_probs_0': probs[:,0],
            'base_probs_1': probs[:,1],
        })
        
        chunk_df = bert(chunk_df)
        f1_bert = f1_score(chunk_y, chunk_df["bert_preds"], average=modelconf.f1_avg)
        chunk_df['base_f1'] = f1_base
        chunk_df['bert_f1'] = f1_bert

        write_mode = 'replace' if i == 0 else 'append'
        chunk_df.to_sql('X_test', conn, if_exists=write_mode, index=False)
        print(f"[{end_idx}/{total_samples}] ...")
        
    conn.close()
    print("Completed!")

def create_review_table(conn):
    margin_threshold = 0.2

    query_1 = "SELECT * FROM X_test WHERE base_pred != bert_preds"
    df_disagree = pd.read_sql(query_1, conn)
    df_disagree.to_sql("prediction_unmatch", conn, if_exists="replace", index=False)

    query_2 = f"""
                SELECT * FROM X_test
                WHERE ABS(base_probs_0 - base_probs_1) < {margin_threshold}
                OR ABS(scores_0 - scores_1) < {margin_threshold}
            """
    df_ambiguous = pd.read_sql(query_2, conn)
    df_ambiguous.to_sql("ambiguous_confidence", conn, if_exists="replace", index=False)

    query_3 = """
                SELECT * FROM X_test 
                WHERE base_pred = bert_preds 
                AND base_pred != label"""

    df_fooled = pd.read_sql(query_3, conn)
    df_fooled.to_sql("label_unmatch", conn, if_exists="replace", index=False)
    print("Completed!")
    conn.close()

def read_table(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print("tables:", tables)


if __name__ == "__main__":
    db_path = paths.DB_PATH / "explorations.db"
    conn = sqlite3.connect(db_path)
    create_review_table(conn)
    # predict_to_db_by_batch()

    wrong_ids = ["2177790500", "1979097603"]
    difficult_to_learn = ["1997786292", "1989335189"]
    
