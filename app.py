from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# 1. PAGE CONFIG (must be first Streamlit call)
# -------------------------------
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="centered",
)

# -------------------------------
# 2. LIGHT STYLING (works in light and dark themes)
# -------------------------------
st.markdown(
    """
    <style>
    h1 { text-align: center; }
    .movie-card {
        background-color: rgba(128, 128, 128, 0.15);
        border-left: 4px solid #ff4b6e;
        padding: 14px 18px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .movie-card h4 { margin: 0 0 6px 0; }
    .movie-card p  { margin: 0; opacity: 0.85; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------
# 3. LOAD DATA
# -------------------------------
DATA_FILE = Path(__file__).parent / "movies.csv"


@st.cache_data
def load_data() -> pd.DataFrame:
    data = pd.read_csv(DATA_FILE)
    data = data.drop_duplicates(subset="movieId")
    data = data.drop_duplicates(subset="title")
    data = data.dropna(subset=["title"]).reset_index(drop=True)

    data["genres"] = data["genres"].fillna("").str.replace("|", " ", regex=False)
    data["clean_title"] = data["title"].str.replace(
        r"\s*\(\d{4}\)\s*$", "", regex=True
    )
    data["combined_features"] = data["clean_title"] + " " + data["genres"]
    return data


# -------------------------------
# 4. BUILD MODEL
# -------------------------------
@st.cache_resource
def build_model(features: tuple):
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(pd.Series(features))
    return matrix


# -------------------------------
# 5. RECOMMENDATION FUNCTION
# -------------------------------
def recommend_movies(movies, tfidf_matrix, movie_title, number=10):
    matches = movies.index[movies["title"] == movie_title]
    if len(matches) == 0:
        return []

    movie_index = matches[0]
    scores = cosine_similarity(tfidf_matrix[movie_index], tfidf_matrix).flatten()

    results = []
    for index in scores.argsort()[::-1]:
        if index == movie_index:
            continue
        results.append(
            {
                "title": movies.at[index, "title"],
                "genres": movies.at[index, "genres"],
                "score": round(float(scores[index]), 3),
            }
        )
        if len(results) == number:
            break
    return results


# -------------------------------
# 6. PAGE
# -------------------------------
st.title("🎬 Movie Recommendation System")
st.caption("Discover movies similar to your favourites!")
st.divider()

try:
    with st.spinner("Loading movies and building the model..."):
        movies = load_data()
        tfidf_matrix = build_model(tuple(movies["combined_features"]))
except Exception as e:
    st.error("The app could not start.")
    st.exception(e)
    st.stop()

st.subheader("🔍 Find Your Next Movie")

selected_movie = st.selectbox(
    "Search or select a movie",
    options=movies["title"].tolist(),
    index=0,
)

number = st.slider("Number of recommendations", 5, 10, 10)

if st.button("✨ Recommend Movies", use_container_width=True):
    results = recommend_movies(movies, tfidf_matrix, selected_movie, number)

    if not results:
        st.error("No recommendations found.")
    else:
        st.success(f"Movies similar to {selected_movie}")
        st.subheader("🍿 Recommended Movies")

        for i, movie in enumerate(results, 1):
            st.markdown(
                f"""
                <div class="movie-card">
                    <h4>{i}. {escape(movie['title'])}</h4>
                    <p>🎭 Genres: {escape(movie['genres'])}</p>
                    <p>⭐ Similarity score: {movie['score']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

st.divider()
st.caption("Built with Python, Machine Learning and Streamlit ❤️")
