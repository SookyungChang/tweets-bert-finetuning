import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv
from googleapiclient.discovery import build
import pandas as pd
from src.data.preprocess import remove_urls, safe_detect
from config import PathConfig, YoutubeCommentsConfig

load_dotenv()


def get_comments(video_id, max_results, max_pages):
    # Load the YouTube API key from environment variables (keeps it secure)
    api_key = os.getenv("YOUTUBE_API_KEY")
    # Build the YouTube API client using the google-api-python-client library
    youtube = build("youtube", "v3", developerKey=api_key)

    comments = []  # Will hold all collected comment data
    next_page_token = None  # Used to navigate through paginated API results
    page_count = 0  # Tracks how many pages we've fetched so far

    while True:
        # Build the API request to fetch comment threads for the given video
        request = youtube.commentThreads().list(
            part="snippet",  # "snippet" includes the actual comment content
            videoId=video_id,  # The ID of the YouTube video to fetch comments from
            maxResults=max_results,  # How many comments to fetch per page (default 100)
            pageToken=next_page_token,  # None on first run; set to next page after that
            textFormat="plainText",  # Return plain text instead of HTML-formatted text
        )

        # Execute the request and get the response from YouTube's API
        response = request.execute()

        # Loop through each comment thread returned in this page
        for item in response["items"]:
            # Drill into the nested response structure to get the comment details
            comment = item["snippet"]["topLevelComment"]["snippet"]

            # Store only the fields we care about
            comments.append(
                {
                    "author": comment["authorDisplayName"],
                    "text": comment["textDisplay"],  # The comment text
                    "likeCount": comment["likeCount"],  # Number of likes on the comment
                    "publishedAt": comment["publishedAt"],  # When posted
                    "commentId": item["id"],  # Unique ID for the comment thread
                }
            )

        page_count += 1  # Increment page counter after processing each page

        # Get the token for the next page (will be None if this is the last page)
        next_page_token = response.get("nextPageToken")

        # Stop if there are no more pages, or we've hit our page limit
        if not next_page_token or page_count >= max_pages:
            break

    # Return all collected comments as a pandas DataFrame for easy analysis
    return pd.DataFrame(comments)


def filter_english_comments(df):
    """Filter a DataFrame down to high-confidence English comments only."""

    # Step 1: Remove URLs from all comments before processing
    df["text"] = df["text"].apply(remove_urls)

    # Step 2: Only run language detection on comments longer than 5 characters
    # (avoids wasting API calls on very short strings)
    mask = df["text"].str.len() > 5

    # Step 3: Apply safe_detect only to the rows that pass the length check
    df.loc[mask, "language"] = df.loc[mask, "text"].apply(safe_detect)

    # Step 4: Drop rows where language is null (non-English, low confidence, or too short)
    return df[df["language"].notnull()].reset_index(drop=True)


def comments_save(results):
    paths = PathConfig
    results.to_csv(
        paths.DATA_PATH
        / f"youtube_comments/{video_id}_{YoutubeCommentsConfig.MAX_RESULTS}_{YoutubeCommentsConfig.MAX_PAGES}.csv",
        index=False,
    )
    print(
        f"Saved {len(results)} English comments to {paths.DATA_PATH / f'youtube_comments/{video_id}_{YoutubeCommentsConfig.MAX_RESULTS}_{YoutubeCommentsConfig.MAX_PAGES}.csv'}"
    )


if __name__ == "__main__":
    video_id = YoutubeCommentsConfig.VIDEO_ID
    comments_df = get_comments(
        video_id, YoutubeCommentsConfig.MAX_RESULTS, YoutubeCommentsConfig.MAX_PAGES
    )
    english_comments_df = filter_english_comments(comments_df)
    comments_save(english_comments_df)
