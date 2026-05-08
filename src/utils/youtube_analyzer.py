from urllib.parse import urlparse, parse_qs  # ← add this at the top

def extract_video_id(url_or_id: str) -> str:
    """Extract video ID from a YouTube URL or return as-is if already an ID."""
    url_or_id = url_or_id.strip()
    
    # If it's already just an ID (no slashes or dots) return directly
    if "youtube.com" not in url_or_id and "youtu.be" not in url_or_id:
        return url_or_id
    
    # Handle youtu.be/VIDEO_ID format
    if "youtu.be" in url_or_id:
        return url_or_id.split("youtu.be/")[-1].split("?")[0]
    
    # Handle youtube.com/watch?v=VIDEO_ID format
    parsed = urlparse(url_or_id)
    params = parse_qs(parsed.query)
    if "v" in params:
        return params["v"][0]
    
    raise ValueError("Could not extract video ID from URL")