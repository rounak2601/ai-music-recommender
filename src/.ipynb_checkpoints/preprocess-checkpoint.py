 # preprocess.py

import os
import pandas as pd
import re
import nltk
import joblib
import logging
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------- SETUP ---------------- #

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("preprocess.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logging.info("🚀 Starting preprocessing...")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "spotify_millsongdata.csv")

# ---------------- NLTK ---------------- #

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# ---------------- LOAD DATA ---------------- #

try:
    df = pd.read_csv(DATA_PATH)
    logging.info("✅ Dataset loaded: %d rows", len(df))
except Exception as e:
    logging.error("❌ Failed to load dataset: %s", str(e))
    raise e

# Drop unnecessary column
df = df.drop(columns=['link'], errors='ignore').reset_index(drop=True)

# ---------------- TEXT CLEANING ---------------- #

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", str(text))
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

logging.info("🧹 Cleaning text...")
df['cleaned_text'] = df['text'].apply(preprocess_text)
logging.info("✅ Text cleaned.")

# ---------------- TF-IDF ---------------- #

logging.info("🔠 Vectorizing using TF-IDF...")
tfidf = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf.fit_transform(df['cleaned_text'])
logging.info("✅ TF-IDF matrix shape: %s", tfidf_matrix.shape)

# ---------------- SAVE FILES ---------------- #

logging.info("💾 Saving processed files...")

joblib.dump(df, os.path.join(BASE_DIR, 'df_cleaned.pkl'))
joblib.dump(tfidf, os.path.join(BASE_DIR, 'tfidf_vectorizer.pkl'))
joblib.dump(tfidf_matrix, os.path.join(BASE_DIR, 'tfidf_matrix.pkl'))

logging.info("✅ All files saved successfully.")
logging.info("🎉 Preprocessing complete.")