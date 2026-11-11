
import streamlit as st
import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# 1. PAGE CONFIGURATION
# -------------------------------

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# -------------------------------
# 2. CUSTOM DESIGN
# -------------------------------

st.markdown("""
<style>
.stApp {
    background-color: #10101a;
    color: white;
}
h1 {
    color: #ff4b6e;
    text-align: center;
}
.movie-card {
    background-color: #202033;
    padding: 18px;
    border-radius: 12px;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# 3. LOAD MOVIE DATA
# -------------------------------

@st.cache_data
def load_data():
    data = pd.read_csv("movies.csv")

    data = data.drop_duplicates(subset="movieId").copy()
    data = data.reset_index(drop=True)

    data["clean_title"] = data["title"].str.replace(
        r"\s*\(\d{4}\)\s*$", "", regex=True
    )

    data["genres"] = data["genres"].fillna(
        ""
    ).str.replace("|", " ", regex=False)

    data["combined_features"] = (
        data["clean_title"] + " " + data["genres"]
    )

    return data

# -------------------------------
# 4. BUILD ML MODEL
# -------------------------------

@st.cache_resource
def build_model():
    data = load_data()

    vectorizer = TfidfVectorizer(stop_words="english")

    matrix = vectorizer.fit_transform(
        data["combined_features"]
    )

    return vectorizer, matrix

movies = load_data()
vectorizer, tfidf_matrix = build_model()

# -------------------------------
# 5. RECOMMENDATION FUNCTION
# -------------------------------

def recommend_movies(movie_name, number=10):

    # Match the complete movie title
    matches = movies[
        movies["title"].str.casefold() == movie_name.casefold()
    ]

    if matches.empty:
        return None, []

    movie_index = matches.index[0]

    # Calculate similarity
    scores = cosine_similarity(
        tfidf_matrix[movie_index],
        tfidf_matrix
    ).flatten()

    similar_indices = scores.argsort()[::-1]

    recommendations = []

    for index in similar_indices:

        if index == movie_index:
            continue

        recommendations.append({
            "title": movies.iloc[index]["title"],
            "genres": movies.iloc[index]["genres"],
            "score": round(float(scores[index]), 3)
        })

        if len(recommendations) == number:
            break

    return movies.iloc[movie_index]["title"], recommendations


