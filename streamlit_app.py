import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="🎵 Hindi Song Recommender", layout="centered")

st.title("🎶 Hindi Song Recommender App")
st.write("Get similar songs based on genre, artist, and movie 🎧")

# Load the dataset
df = pd.read_csv("ex.csv", encoding="ISO-8859-1")

# Combine relevant features for recommendation
def combine_features(row):
    return f"{row['Singer/Artists']} {row['Genre']} {row['Album/Movie']}"

df["combined"] = df.apply(combine_features, axis=1)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df["combined"])

# Similarity matrix
similarity = cosine_similarity(tfidf_matrix)

# Song selection
song_list = df["Song-Name"].tolist()
selected_song = st.selectbox("🎵 Choose a song you like:", song_list)

# Recommend button
if st.button("Recommend Similar Songs"):
    idx = df[df["Song-Name"] == selected_song].index[0]
    similar_songs = list(enumerate(similarity[idx]))
    similar_songs = sorted(similar_songs, key=lambda x: x[1], reverse=True)[1:6]

    st.subheader("🎧 You might also enjoy:")
    for i in similar_songs:
        row = df.iloc[i[0]]
        st.markdown(f"**🎵 {row['Song-Name']}** — *{row['Singer/Artists']}*  \n🎬 *Movie:* {row['Album/Movie']} | ⭐ *Rating:* {row['User-Rating']}")
