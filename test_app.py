from src.inference import predictor_bert
from huggingface_hub import snapshot_download
import pandas as pd
from src.config import PathConfig, YoutubeCommentsConfig

def app():

    bert_path = snapshot_download(repo_id="sweetguma/bert-sentiment-model")
    bert = predictor_bert.Predictor(bert_path)

    paths = PathConfig()

    csv_filename = f"{YoutubeCommentsConfig.VIDEO_ID}_{YoutubeCommentsConfig.MAX_RESULTS}_{YoutubeCommentsConfig.MAX_PAGES}.csv"
    df = pd.read_csv(paths.YOUTUBE_COMMENTS_PATH / csv_filename)
    df_predicted = bert.predict_df(df)
    print(df_predicted.head())

if __name__ == "__main__":
    app()