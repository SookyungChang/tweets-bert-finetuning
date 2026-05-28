import os
import json
from dotenv import load_dotenv
load_dotenv()
from groq import Groq
import pandas as pd

def label_by_qroq(df: pd.DataFrame) -> pd.DataFrame:
    client = Groq()

    system_prompt = """
    You are an expert in sentiment analysis. Your task is to analyze the sentiment of the provided comments.
    For each comment, determine if the sentiment is 'positive', 'negative', or 'neutral'.
    Analyze the text and determine the user's underlying emotional state.
    [Strict Rules]
    1. Watch out for sarcasm and irony. (e.g., "Yeah, great job making it worse" is NEGATIVE).
    2. Respond ONLY with a valid JSON object in the following format. Do not include any conversational text outside the JSON.
    [Expected Output Format]
    {
      "results": [
        {"id": 1, "sentiment": "positive", "reason": "...brief reason..."},
        {"id": 2, "sentiment": "negative", "reason": "...brief reason..."}
      ]
    }
    """

    # Build input from df using commentId as id
    user_content = "Analyze the sentiment for these comments:\n"
    for _, row in df.iterrows():
        user_content += f"[ID: {row['commentId']}] \"{row['text']}\"\n"

    sentiment_map = {
        "positive": 1,
        "negative": 0,
        "neutral": -1
    }

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            response_format={"type": "json_object"},
            temperature=0.2
        )

        raw_json_output = response.choices[0].message.content
        parsed_data = json.loads(raw_json_output)

        # Save to JSON file
        with open("sentiment_results.json", "w", encoding="utf-8") as f:
            json.dump(parsed_data, f, indent=2, ensure_ascii=False)
        print("'sentiment_results.json' saved.")

        # Read back JSON and build result df
        with open("sentiment_results.json", "r", encoding="utf-8") as f:
            loaded = json.load(f)

        result_rows = []
        for item in loaded["results"]:
            result_rows.append({
                "commentId": item["id"],
                "groq": sentiment_map.get(item["sentiment"], -2)  # -2 if unexpected value
            })

        result_df = pd.DataFrame(result_rows)

        # Merge back onto original df on commentId
        output_df = df.merge(result_df, on="commentId", how="left")
        output_df["groq"] = output_df["groq"].fillna(-2).astype(int)  # -2 for any missing

        return output_df

    except Exception as e:
        print(f"ERROR: {e}")
        # Return original df with all -2 (error) on failure
        df["groq"] = -2
        return df