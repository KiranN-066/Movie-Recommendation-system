
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

    matches = movies[
        movies["clean_title"].str.contains(
            re.escape(movie_name),
            case=False,
            na=False
        )
    ]

    if matches.empty:
        return None, []

    movie_index = matches.index[0]

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

# -------------------------------
# 6. WEBSITE HEADER
# -------------------------------

st.markdown(
    "<h1>🎬 Movie Recommendation System</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;'>"
    "Discover movies similar to your favourites!"
    "</p>",
    unsafe_allow_html=True
)

st.divider()

# -------------------------------
# 7. MOVIE SEARCH
# -------------------------------

st.subheader("🔍 Find Your Next Movie")

selected_movie = st.selectbox(
    "Search or select a movie",
    options=movies["title"].tolist()
)

number = st.slider(
    "Number of recommendations",
    min_value=5,
    max_value=10,
    value=10
)

# -------------------------------
# 8. RECOMMENDATION BUTTON
# -------------------------------

if st.button("✨ Recommend Movies", use_container_width=True):

    selected_title, results = recommend_movies(
        selected_movie,
        number
    )

    if not results:
        st.error("No recommendations found.")

    else:
        st.success(f"Movies similar to {selected_title}")

        st.subheader("🍿 Recommended Movies")

        for i, movie in enumerate(results, 1):

            st.markdown(
                f"""
                <div class="movie-card">
                    <h3>{i}. {movie['title']}</h3>
                    <p>🎭 Genres: {movie['genres']}</p>
                    <p>⭐ Similarity Score: {movie['score']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

st.divider()

st.markdown(
    "<p style='text-align:center;'>"
    "Built with Python, Machine Learning and Streamlit ❤️"
    "</p>",
    unsafe_allow_html=True
)
