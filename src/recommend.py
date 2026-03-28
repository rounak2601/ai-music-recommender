import os
import joblib
import requests
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PATHS ----------------

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PKL_DF     = os.path.join(BASE_DIR, "df_cleaned.pkl")
PKL_TFIDF  = os.path.join(BASE_DIR, "tfidf_matrix.pkl")
CSV_PATH   = os.path.join(BASE_DIR, "spotify_millsongdata.csv")

# ---------------- BUILD PKL IF NOT EXISTS ----------------

def build_pkl_files():
    nltk.download('punkt',      quiet=True)
    nltk.download('stopwords',  quiet=True)
    nltk.download('punkt_tab',  quiet=True)
    stop_words = set(stopwords.words('english'))

    def clean(text):
        text = re.sub(r"[^a-zA-Z\s]", "", str(text)).lower()
        tokens = word_tokenize(text)
        return " ".join([w for w in tokens if w not in stop_words])

    df = pd.read_csv(CSV_PATH)
    df = df.drop(columns=['link'], errors='ignore').reset_index(drop=True)
    df['cleaned_text'] = df['text'].apply(clean)

    tfidf  = TfidfVectorizer(max_features=5000)
    matrix = tfidf.fit_transform(df['cleaned_text'])

    joblib.dump(df,     PKL_DF)
    joblib.dump(matrix, PKL_TFIDF)
    return df, matrix

if os.path.exists(PKL_DF) and os.path.exists(PKL_TFIDF):
    df           = joblib.load(PKL_DF)
    tfidf_matrix = joblib.load(PKL_TFIDF)
else:
    df, tfidf_matrix = build_pkl_files()

# ---------------- LASTFM API ----------------

API_KEY  = "a78b218463cf09dc9760ee564d294766"
BASE_URL = "http://ws.audioscrobbler.com/2.0/"

# ---------------- GET ARTIST IMAGE ----------------

def get_artist_image(artist):
    try:
        params = {
            "method":  "artist.getInfo",
            "api_key": API_KEY,
            "artist":  artist,
            "format":  "json"
        }
        response = requests.get(BASE_URL, params=params, timeout=5)
        data     = response.json()
        images   = data.get("artist", {}).get("image", [])
        for img in reversed(images):
            url = img.get("#text", "")
            if url and url != "":
                return url
        return None
    except Exception:
        return None

# ---------------- GET SONG DETAILS ----------------

def get_song_details(song, artist):
    try:
        params = {
            "method":  "track.getInfo",
            "api_key": API_KEY,
            "artist":  artist,
            "track":   song,
            "format":  "json"
        }
        response = requests.get(BASE_URL, params=params, timeout=5)
        data     = response.json()

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

        if not image:
            image = get_artist_image(artist)

        return image, link

    except Exception:
        return get_artist_image(artist), None

# ---------------- RECOMMEND SONGS ----------------

def recommend_songs(song_name, top_n=10, mood_keywords=None):
    idx = df[df["song"].str.lower() == song_name.lower()].index

    if len(idx) == 0:
        return None

    idx = idx[0]

    sim_scores = cosine_similarity(
        tfidf_matrix[idx],
        tfidf_matrix
    ).flatten()

    sim_scores = sorted(enumerate(sim_scores), key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:]

    if mood_keywords:
        text_col = "cleaned_text" if "cleaned_text" in df.columns else "text"
        filtered = []
        for i, score in sim_scores:
            lyric = str(df.iloc[i].get(text_col, "")).lower()
            if any(kw in lyric for kw in mood_keywords):
                filtered.append((i, score))
        candidates = filtered if len(filtered) >= top_n else sim_scores
    else:
        candidates = sim_scores

    candidates = candidates[:top_n]

    recommendations = []
    for i, score in candidates:
        song   = df.iloc[i]["song"]
        artist = df.iloc[i]["artist"]
        image, link = get_song_details(song, artist)

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

# ---------------- TRENDING SONGS ----------------

def get_trending_songs(limit=10):
    try:
        params = {
            "method":  "chart.getTopTracks",
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

            image, detail_link = get_song_details(song, artist)

            if not link and detail_link:
                link = detail_link

            if not link:
                link = f"https://www.last.fm/music/{artist.replace(' ', '+')}/{song.replace(' ', '+')}"

            trending.append({
                "song":      song,
                "artist":    artist,
                "image":     image,
                "link":      link,
                "listeners": f"{listeners:,}"
            })

        return trending

    except Exception:
        return []