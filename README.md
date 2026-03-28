# 🎶 AI Music Recommender

A Machine Learning web application that recommends similar songs based on lyrics using **TF-IDF Vectorization** and **Cosine Similarity**, with mood/genre filters, a playlist generator, and real-time trending songs powered by the **Last.fm API** — deployed with **Streamlit**.

🔗 **Live Web App:** [🚀 Click Here](https://rounak2601-ai-music-recommender-srcmain-gxl1lh.streamlit.app)

---

## 🚀 Project Overview

This project uses **Natural Language Processing (NLP)** techniques to analyze song lyrics and recommend similar songs. Users can filter recommendations by mood and genre, generate personalized playlists, and discover what's trending globally — all in real time.

---

## 🧠 Machine Learning Details

- **Technique Used:** TF-IDF Vectorization + Cosine Similarity
- **Type:** Content-Based Filtering (NLP)
- **Key Features Used:**
  - Song Lyrics (cleaned & preprocessed)
  - Artist Name
  - Mood Keywords
  - Genre Keywords
- **Model Files:**
  - `src/tfidf_vectorizer.pkl`
  - `src/tfidf_matrix.pkl`
  - `src/df_cleaned.pkl`

The model development and notes are available in the preprocessing script.

---

## ✨ App Features

- 🎵 **Song Recommendations** — Pick any song and get 10 similar songs using lyric-based ML
- 🎛️ **Filters** — Filter by Genre (Pop, Rock, Hip-Hop, Country, Classical) and Artist
- 🎧 **Playlist Generator** — Generate a custom playlist filtered by Mood + Genre
- 🔥 **Trending Songs** — Real-time globally trending songs via Last.fm chart API
- 🖼️ **Album Art** — Fetches album/artist images from Last.fm API
- ▶️ **Listen Links** — Direct links to Last.fm for every song

---

## 🛠️ Tech Stack

- Python 3.x
- Pandas
- Scikit-learn
- NLTK
- Streamlit
- Joblib
- Last.fm API
- Git & GitHub
- Streamlit Cloud (Deployment)

---

## 📁 Repository Structure

```
ai-music-recommender/
│
├── main.py                  ← Streamlit app (UI + logic)
├── requirements.txt         ← Python dependencies
├── start_app.txt            ← Steps to run the app
│
└── src/
    ├── preprocess.py        ← Data cleaning & TF-IDF vectorization
    ├── recommend.py         ← Recommendation & Last.fm API logic
    ├── spotify_millsongdata.csv  ← Dataset (add manually)
    ├── tfidf_vectorizer.pkl ← Saved TF-IDF vectorizer (after preprocessing)
    ├── tfidf_matrix.pkl     ← Saved TF-IDF matrix (after preprocessing)
    └── df_cleaned.pkl       ← Cleaned DataFrame (after preprocessing)
```

---

## ⚙️ Installation & Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/YourUsername/ai-music-recommender.git
cd ai-music-recommender
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

Download the `spotify_millsongdata.csv` dataset and place it inside the `src/` folder.

### 5. Run preprocessing (generates .pkl files)

```bash
cd src
python preprocess.py
```

### 6. Run the Streamlit app

```bash
cd ..
streamlit run main.py
```

---

## 🌐 Deployment

This application is deployed on **Streamlit Cloud**.

- The deployment automatically installs dependencies from `requirements.txt`
- Preprocessing must be run once locally to generate the `.pkl` files, which are then committed to the repo

---

## 🚀 Future Enhancements

- Add Spotify API integration for richer song data
- Use deep learning embeddings (Word2Vec / BERT) for better recommendations
- Add user login and save favourite songs
- Add audio preview clips
- Better UI / UX with custom Streamlit components

---

## 👤 Author

**Rounak Tilante**
Aspiring Data Analyst | ML Enthusiast
B.Tech Computer Science & Engineering (2025)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
