import os
import joblib
import requests
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- LOAD DATA ----------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

df           = joblib.load(os.path.join(BASE_DIR, "df_cleaned.pkl"))
tfidf_matrix = joblib.load(os.path.join(BASE_DIR, "tfidf_matrix.pkl"))

# ---------------- LASTFM API ----------------

API_KEY  = "a78b218463cf09dc9760ee564d294766"
BASE_URL = "http://ws.audioscrobbler.com/2.0/"

# ---------------- GET ARTIST IMAGE FROM LASTFM ----------------

def get_artist_image(artist):
    """Fetch artist image from Last.fm as fallback when album art is unavailable."""
    try:
        params = {
            "method": "artist.getInfo",
            "api_key": API_KEY,
            "artist": artist,
            "format": "json"
        }
        response = requests.get(BASE_URL, params=params, timeout=5)
        data = response.json()
        images = data.get("artist", {}).get("image", [])
        for img in reversed(images):
            url = img.get("#text", "")
            if url and url != "":
                return url
        return None
    except Exception:
        return None


# ---------------- GET SONG DETAILS FROM LASTFM ----------------

def get_song_details(song, artist):
    """
    Fetch album art and Last.fm page link for a given song.
    Priority: track album art → artist image → None
    """
    try:
        params = {
            "method": "track.getInfo",
            "api_key": API_KEY,
            "artist": artist,
            "track": song,
            "format": "json"
        }
        response = requests.get(BASE_URL, params=params, timeout=5)
        data = response.json()

        image = None
        link  = None

        if "track" in data:
            if "album" in data["track"]:
                images = data["track"]["album"]["image"]
                for img in reversed(images):
                    url = img.get("#text", "")
                    if url and url != "":
                        image = url
                        break
            link = data["track"].get("url", None)

        # Fallback to artist image if album art not found
        if not image:
            image = get_artist_image(artist)

        return image, link

    except Exception:
        return get_artist_image(artist), None


# ---------------- RECOMMEND SONGS ----------------

def recommend_songs(song_name, top_n=10, mood_keywords=None):
    """
    Recommend songs similar to song_name using TF-IDF cosine similarity.

    Parameters:
        song_name     : str  — name of the input song
        top_n         : int  — number of recommendations to return
        mood_keywords : list — optional list of mood keywords to filter results
                               (filters on cleaned_text column)

    Returns:
        list of dicts with keys: song, artist, image, link
    """
    # Find the song index
    idx = df[df["song"].str.lower() == song_name.lower()].index

    if len(idx) == 0:
        return None

    idx = idx[0]

    # Compute cosine similarity between selected song and all songs
    sim_scores = cosine_similarity(
        tfidf_matrix[idx],
        tfidf_matrix
    ).flatten()

    # Sort by similarity score descending, skip index 0 (the song itself)
    sim_scores = sorted(enumerate(sim_scores), key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:]  # remove the input song itself

    # If mood filter is applied, filter candidates whose lyrics contain mood keywords
    if mood_keywords:
        text_col = "cleaned_text" if "cleaned_text" in df.columns else "text"
        filtered = []
        for i, score in sim_scores:
            lyric = str(df.iloc[i].get(text_col, "")).lower()
            if any(kw in lyric for kw in mood_keywords):
                filtered.append((i, score))
        # Fall back to unfiltered if mood filter returns too few results
        candidates = filtered if len(filtered) >= top_n else sim_scores
    else:
        candidates = sim_scores

    # Take top_n candidates
    candidates = candidates[:top_n]

    # Build recommendations list with Last.fm data
    recommendations = []
    for i, score in candidates:
        song   = df.iloc[i]["song"]
        artist = df.iloc[i]["artist"]
        image, link = get_song_details(song, artist)

        # Fallback Last.fm link if API didn't return one
        if not link:
            link = f"https://www.last.fm/music/{artist.replace(' ', '+')}/{song.replace(' ', '+')}"

        recommendations.append({
            "song":   song,
            "artist": artist,
            "image":  image,
            "link":   link,
            "score":  round(float(score), 4)
        })

    return recommendations


# ---------------- REAL-TIME TRENDING SONGS ----------------

def get_trending_songs(limit=10):
    """
    Fetch real-time globally trending songs from Last.fm chart.getTopTracks API.
    Album art is fetched via track.getInfo for each song since chart.getTopTracks
    does not reliably return images.

    Returns:
        list of dicts with keys: song, artist, image, link, listeners
    """
    try:
        params = {
            "method": "chart.getTopTracks",
            "api_key": API_KEY,
            "format":  "json",
            "limit":   limit
        }
        response = requests.get(BASE_URL, params=params, timeout=7)
        data     = response.json()

        tracks   = data.get("tracks", {}).get("track", [])
        trending = []

        for track in tracks:
            song      = track.get("name", "Unknown")
            artist    = track.get("artist", {}).get("name", "Unknown")
            link      = track.get("url", "")
            listeners = int(track.get("listeners", 0))

            # chart.getTopTracks images are always empty — use track.getInfo instead
            image, detail_link = get_song_details(song, artist)

            # Use detail_link if chart didn't give us one
            if not link and detail_link:
                link = detail_link

            # Final fallback link
            if not link:
                link = f"https://www.last.fm/music/{artist.replace(' ', '+')}/{song.replace(' ', '+')}"

            trending.append({
                "song":      song,
                "artist":    artist,
                "image":     image,   # now properly fetched via track.getInfo
                "link":      link,
                "listeners": f"{listeners:,}"
            })

        return trending

    except Exception:
        return []