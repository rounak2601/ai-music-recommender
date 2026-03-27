import streamlit as st
from recommend import df, recommend_songs, get_trending_songs

# ---------------- PAGE SETTINGS ----------------

st.set_page_config(
    page_title="AI Music Recommender",
    page_icon="🎧",
    layout="wide"
)

# ---------------- CSS ----------------

st.markdown("""
<style>

body {
    background-color: #0e1117;
}

.main-title {
    text-align: center;
    color: #1DB954;
    font-size: 50px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: #b3b3b3;
    font-size: 18px;
    margin-bottom: 30px;
}

/* ---- Song Card ---- */
.song-card {
    background-color: #181818;
    padding: 15px;
    border-radius: 15px;
    text-align: center;
    transition: transform 0.3s ease, box-shadow 0.3s ease, background-color 0.3s ease;
    border: 1px solid #282828;
    margin-bottom: 10px;
}

.song-card:hover {
    transform: scale(1.06);
    box-shadow: 0px 8px 25px rgba(29, 185, 84, 0.35);
    background-color: #1f1f1f;
    border: 1px solid #1DB954;
}

.song-card img {
    border-radius: 10px;
    width: 100%;
}

/* ---- Headphone Placeholder ---- */
.placeholder-box {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 10px;
    width: 100%;
    aspect-ratio: 1 / 1;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 52px;
    border: 1px solid #1DB954;
    margin-bottom: 4px;
}

.song-title {
    color: white;
    font-size: 15px;
    font-weight: bold;
    margin-top: 10px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.artist {
    color: #b3b3b3;
    font-size: 13px;
    margin-top: 4px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

/* ---- Listen Button ---- */
.listen-btn {
    background-color: #1DB954;
    color: black !important;
    padding: 8px 18px;
    border-radius: 20px;
    text-decoration: none !important;
    font-size: 13px;
    font-weight: bold;
    display: inline-block;
    margin-top: 10px;
    transition: background-color 0.2s ease, transform 0.2s ease;
}

.listen-btn:hover {
    background-color: #17a348;
    transform: scale(1.05);
    color: black !important;
}

/* ---- Section Divider ---- */
.section-divider {
    border: none;
    border-top: 1px solid #282828;
    margin: 30px 0;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HELPER: Song Card HTML ----------------

def song_card_html(img_url, song_name, artist_name, listen_url):
    """Renders a song card. Shows 🎧 headphone placeholder if no image available."""
    if img_url:
        image_html = f'<img src="{img_url}" alt="cover"/>'
    else:
        image_html = '<div class="placeholder-box">🎧</div>'

    return f'''
    <div class="song-card">
        {image_html}
        <div class="song-title">{song_name}</div>
        <div class="artist">{artist_name}</div>
        <a href="{listen_url}" target="_blank" class="listen-btn">▶ Listen</a>
    </div>
    '''

# ---------------- KEYWORD MAPS ----------------

MOOD_KEYWORDS = {
    "😊 Happy":  ["happy", "joy", "sunshine", "smile", "good", "fun", "dance", "celebrate"],
    "😢 Sad":    ["sad", "cry", "tears", "alone", "miss", "pain", "broken", "lost"],
    "😌 Chill":  ["relax", "easy", "slow", "calm", "peace", "breeze", "night", "dream"],
    "🎉 Party":  ["party", "dance", "night", "loud", "bass", "move", "energy", "wild"],
}

GENRE_KEYWORDS = {
    "All Genres": [],
    "Pop":        ["swift", "bieber", "ariana", "rihanna", "beyonce", "selena", "katy"],
    "Rock":       ["acdc", "metallica", "nirvana", "foo fighters", "queen", "led zeppelin", "guns"],
    "Hip-Hop":    ["eminem", "drake", "kendrick", "jay", "kanye", "tupac", "snoop", "nicki"],
    "Country":    ["strait", "chesney", "urban", "underwood", "jackson", "brooks", "lambert"],
    "Classical":  ["beethoven", "mozart", "bach", "chopin", "vivaldi", "handel"],
}

# ---------------- TITLE ----------------

st.markdown('<div class="main-title">🎶 AI Music Recommender</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Discover similar songs using Machine Learning</div>', unsafe_allow_html=True)

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

# ================================================================
# SECTION 1 — SONG SELECTOR & RECOMMENDATIONS
# ================================================================

st.markdown("### 🎵 Pick a Song & Get Recommendations")

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.markdown("**🎸 Filter by Genre**")
    selected_genre_rec = st.selectbox(
        "Genre", list(GENRE_KEYWORDS.keys()),
        label_visibility="collapsed", key="genre_rec"
    )

# Apply genre filter first to get artist list
song_df = df.copy()
if selected_genre_rec != "All Genres":
    kws  = GENRE_KEYWORDS[selected_genre_rec]
    mask = song_df['artist'].str.lower().apply(lambda a: any(k in a for k in kws))
    genre_filtered_df = song_df[mask] if mask.sum() > 0 else song_df
else:
    genre_filtered_df = song_df

with col2:
    st.markdown("**🎤 Filter by Artist**")
    artist_list = ["All Artists"] + sorted(genre_filtered_df['artist'].dropna().unique().tolist())
    selected_artist = st.selectbox(
        "Artist", artist_list,
        label_visibility="collapsed", key="artist_rec"
    )

# Apply artist filter on top of genre filter
if selected_artist != "All Artists":
    filtered_df = genre_filtered_df[genre_filtered_df['artist'] == selected_artist]
    filtered_df = filtered_df if len(filtered_df) > 0 else genre_filtered_df
else:
    filtered_df = genre_filtered_df

song_list     = sorted(filtered_df['song'].dropna().unique())

with col3:
    st.markdown("**🎵 Choose a Song**")
    selected_song = st.selectbox("Song", song_list,
                                 label_visibility="collapsed", key="song_select")

if st.button("🎧 Get Recommendations"):
    results = recommend_songs(selected_song, top_n=10)

    if not results:
        st.warning("No recommendations found for this song.")
    else:
        st.markdown("## 🎵 Recommended Songs")
        cols = st.columns(5)
        for i, song in enumerate(results):
            with cols[i % 5]:
                listen_url = song["link"] or \
                    f"https://www.last.fm/music/{song['artist'].replace(' ', '+')}/{song['song'].replace(' ', '+')}"
                st.markdown(
                    song_card_html(song["image"], song["song"], song["artist"], listen_url),
                    unsafe_allow_html=True
                )

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

# ================================================================
# SECTION 2 — PLAYLIST GENERATOR
# Filters sit directly above the Generate button
# ================================================================

st.markdown("## 🎧 Generate Playlist")

# ---- Filters RIGHT above the playlist ----
st.markdown("#### 🎛️ Playlist Filters")
st.caption("Select Mood and/or Genre — your playlist will be generated based on these")

pcol1, pcol2 = st.columns([1, 1])

with pcol1:
    st.markdown("**🎭 Mood**")
    selected_mood = st.selectbox(
        "Mood", ["All Moods"] + list(MOOD_KEYWORDS.keys()),
        label_visibility="collapsed", key="mood_playlist"
    )

with pcol2:
    st.markdown("**🎸 Genre**")
    selected_genre = st.selectbox(
        "Genre", list(GENRE_KEYWORDS.keys()),
        label_visibility="collapsed", key="genre_playlist"
    )

# Active filter badge
active_filters = []
if selected_mood  != "All Moods":
    active_filters.append(f"Mood: **{selected_mood}**")
if selected_genre != "All Genres":
    active_filters.append(f"Genre: **{selected_genre}**")

if active_filters:
    st.success(f"✅ Active filters — {' + '.join(active_filters)}")
else:
    st.info("💡 No filters selected — playlist will be random. Select Mood/Genre above to personalise!")

playlist_size = st.slider("Select number of songs", 5, 20, 10)

if st.button("Generate Playlist"):

    pool     = df.copy()
    text_col = "cleaned_text" if "cleaned_text" in pool.columns else "text"

    # Step 1: Genre filter
    if selected_genre != "All Genres":
        gkws       = GENRE_KEYWORDS[selected_genre]
        gmask      = pool['artist'].str.lower().apply(lambda a: any(k in a for k in gkws))
        genre_pool = pool[gmask]
        pool       = genre_pool if len(genre_pool) >= playlist_size else pool

    # Step 2: Mood filter on top of genre pool
    if selected_mood != "All Moods":
        mkws      = MOOD_KEYWORDS[selected_mood]
        mmask     = pool[text_col].str.lower().apply(lambda t: any(k in str(t) for k in mkws))
        mood_pool = pool[mmask]
        pool      = mood_pool if len(mood_pool) >= playlist_size else pool

    sample_songs = pool.sample(min(playlist_size, len(pool))).reset_index(drop=True)

    if active_filters:
        label = " + ".join(active_filters).replace("**", "")
        st.markdown(f"### 🎵 Your {label} Playlist")
    else:
        st.markdown("### 🎵 Your Playlist")

    st.caption(f"Generated {len(sample_songs)} songs")

    for i in range(len(sample_songs)):
        s_name = sample_songs.iloc[i]['song']
        a_name = sample_songs.iloc[i]['artist']
        url    = f"https://www.last.fm/music/{a_name.replace(' ', '+')}/{s_name.replace(' ', '+')}"
        st.markdown(
            f'🎵 **{s_name}** — {a_name} &nbsp;&nbsp;'
            f'<a href="{url}" target="_blank" class="listen-btn"'
            f' style="font-size:11px;padding:4px 10px;">▶ Listen</a>',
            unsafe_allow_html=True
        )

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

# ================================================================
# SECTION 3 — TRENDING SONGS (REAL-TIME)
# ================================================================

st.markdown("## 🔥 Trending Songs")
st.caption("Live data from Last.fm global charts")

with st.spinner("Fetching live trending songs..."):
    trending = get_trending_songs(limit=10)

if not trending:
    st.warning("Could not fetch trending songs. Please check your internet connection.")
else:
    cols = st.columns(5)
    for i, song in enumerate(trending):
        with cols[i % 5]:
            listen_url = song["link"] or \
                f"https://www.last.fm/music/{song['artist'].replace(' ', '+')}/{song['song'].replace(' ', '+')}"
            st.markdown(
                song_card_html(song["image"], song["song"], song["artist"], listen_url),
                unsafe_allow_html=True
            )