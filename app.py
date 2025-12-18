import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import ast

st.set_page_config(
    page_title="🎬 Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# ========================================
# Model Definition
# ========================================
class MetaLearner(nn.Module):
    def __init__(self, n_features=8):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

# ========================================
# Load Data and Models
# ========================================
@st.cache_resource
def load_all():
    # Load dataframes
    movies = pd.read_csv('movies_filtered.csv')
    ratings = pd.read_csv('ratings_filtered.csv')
    
    # Load embeddings
    movie_embeddings = np.load('movie_embeddings.npy')
    
    # Load mappings
    with open('mappings.pkl', 'rb') as f:
        mappings = pickle.load(f)

    # Load ALS factors
    als_user_factors = np.load('als_user_factors.npy')
    als_item_factors = np.load('als_item_factors.npy')
    
    # Load PyTorch model
    model = MetaLearner(n_features=8)
    model.load_state_dict(torch.load('meta_learner_weights.pt', map_location='cpu'))
    model.eval()
    
    return movies, ratings, movie_embeddings, mappings, als_user_factors, als_item_factors, model

movies, ratings, movie_embeddings, mappings, als_user_factors, als_item_factors, model = load_all()

# Extract mappings
user_to_idx = mappings['user_to_idx']
movie_to_idx = mappings['movie_to_idx']
movie_id_to_idx = mappings['movie_id_to_idx']
movieId_to_idx = mappings['movieId_to_idx']
user_profiles = mappings['user_profiles']
user_genre_prefs = mappings['user_genre_prefs']
all_genres = mappings['all_genres']

# ========================================
# Helper Functions
# ========================================
def predict_rating(user_id, movie_id):
    user_idx = user_to_idx.get(user_id)
    movie_idx = movie_to_idx.get(movie_id)
    
    if user_idx is None or movie_idx is None:
        return 2.5
    
    if user_idx >= als_user_factors.shape[0] or movie_idx >= als_item_factors.shape[0]:
        return 2.5
    
    score = np.dot(als_user_factors[user_idx], als_item_factors[movie_idx])
    return np.clip(score, 0.5, 5.0)

def parse_genres(genre_str):
    try:
        genres = ast.literal_eval(genre_str)
        return [g['name'] for g in genres]
    except:
        return []

def compute_features(user_id, movie_id):
    features = []
    
    # Feature 1: CF Score
    cf_score = predict_rating(user_id, movie_id) / 5.0
    features.append(cf_score)
    
    movie_row = movies[movies['id'] == movie_id]
    movie_idx = movie_id_to_idx.get(movie_id)
    
    if len(movie_row) > 0 and movie_idx is not None:
        row = movie_row.iloc[0]
        
        # Feature 2: Popularity
        vote_count = row.get('vote_count', 0)
        if pd.isna(vote_count):
            vote_count = 0
        popularity = min(vote_count / 1000, 1.0)
        features.append(popularity)
        
        # Feature 3: Average Rating
        avg_rating = row.get('vote_average', 5.0)
        if pd.isna(avg_rating):
            avg_rating = 5.0
        features.append(avg_rating / 10.0)
        
        # Feature 4: Content Similarity
        if user_id in user_profiles and movie_idx < len(movie_embeddings):
            content_sim = cosine_similarity(
                movie_embeddings[movie_idx].reshape(1, -1),
                user_profiles[user_id].reshape(1, -1)
            )[0][0]
        else:
            content_sim = 0.5
        features.append(content_sim)
        
        # Feature 5: Genre Overlap
        movie_genres = set(parse_genres(str(row.get('genres', '[]'))))
        if user_id in user_genre_prefs and user_genre_prefs[user_id]:
            user_top_genres = set(list(user_genre_prefs[user_id].keys())[:5])
            genre_overlap = len(movie_genres & user_top_genres) / max(len(user_top_genres), 1)
        else:
            genre_overlap = 0.5
        features.append(genre_overlap)
        
        # Feature 6: User Activity
        user_ratings_count = len(ratings[ratings['userId'] == user_id])
        user_activity = min(user_ratings_count / 100, 1.0)
        features.append(user_activity)
        
        # Feature 7: Rating Consistency
        movie_ratings = ratings[ratings['movieId'] == row.get('movieId', -1)]['rating']
        if len(movie_ratings) > 1:
            variance = movie_ratings.var()
            consistency = 1 / (1 + variance)
        else:
            consistency = 0.5
        features.append(consistency)
        
        # Feature 8: Vote Ratio
        if vote_count > 0:
            vote_ratio = avg_rating / 10.0
        else:
            vote_ratio = 0.5
        features.append(vote_ratio)
    else:
        features.extend([0.5] * 7)
    
    return features


def recommend_for_user(user_id, n_recommendations=10):
    user_rated = set(ratings[ratings['userId'] == user_id]['movieId'].tolist())
    scores = []
    
    for idx, row in movies.iterrows():
        movie_id = row['id']
        movielens_id = row['movieId']
        
        if movielens_id in user_rated:
            continue
        
        features = compute_features(user_id, movie_id)
        features_tensor = torch.tensor([features], dtype=torch.float32)
        
        with torch.no_grad():
            score = model(features_tensor).item()
        
        scores.append((row['title'], score, movielens_id, row.get('vote_average', 0), row.get('genres', '[]')))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:n_recommendations]

# ========================================
# Streamlit UI
# ========================================
st.title("🎬 Movie Recommendation System")
st.markdown("*Powered by Hybrid AI: Content + Collaborative Filtering + Neural Network*")

# Sidebar
st.sidebar.header("⚙️ Settings")
n_recs = st.sidebar.slider("Number of recommendations", 5, 20, 10)

# Get valid user IDs
valid_users = sorted(ratings['userId'].unique().tolist())

st.sidebar.header("📊 Dataset Info")
st.sidebar.write(f"Movies: {len(movies)}")
st.sidebar.write(f"Users: {len(valid_users)}")
st.sidebar.write(f"Ratings: {len(ratings)}")

# Main content
tab1, tab2, tab3 = st.tabs(["🎯 Get Recommendations", "🔍 Browse Movies", "📈 Model Info"])

with tab1:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Select User")
        user_id = st.selectbox("Choose User ID", valid_users, index=0)
        
        # Show user's rating history
        user_history = ratings[ratings['userId'] == user_id].merge(
            movies[['movieId', 'title']], on='movieId'
        ).sort_values('rating', ascending=False).head(5)
        
        if len(user_history) > 0:
            st.write("**User's Top Rated Movies:**")
            for _, row in user_history.iterrows():
                st.write(f"⭐ {row['rating']:.1f} - {row['title'][:30]}")
        
        get_recs = st.button("🎬 Get Recommendations", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("Recommendations")
        
        if get_recs:
            with st.spinner("Finding movies you'll love..."):
                recommendations = recommend_for_user(user_id, n_recommendations=n_recs)
            
            if recommendations:
                for i, (title, score, mid, rating, genres) in enumerate(recommendations, 1):
                    with st.container():
                        cols = st.columns([0.5, 3, 1, 1])
                        cols[0].write(f"**#{i}**")
                        cols[1].write(f"**{title}**")
                        cols[2].write(f"⭐ {rating:.1f}")
                        cols[3].progress(score, text=f"{score:.0%}")
                        
                        genre_list = parse_genres(str(genres))
                        if genre_list:
                            st.caption(" | ".join(genre_list[:3]))
                        st.divider()
            else:
                st.warning("No recommendations found for this user.")
        else:
            st.info("👈 Select a user and click 'Get Recommendations'")

with tab2:
    st.subheader("Browse All Movies")
    search = st.text_input("🔍 Search movies", "")
    
    filtered_movies = movies[movies['title'].str.contains(search, case=False, na=False)] if search else movies
    
    st.dataframe(
        filtered_movies[['title', 'vote_average', 'genres']].head(50),
        use_container_width=True,
        hide_index=True
    )

with tab3:
    st.subheader("How It Works")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📝 Content-Based")
        st.write("Analyzes movie descriptions using sentence transformers to find similar movies.")
    
    with col2:
        st.markdown("### 👥 Collaborative")
        st.write("Uses ALS matrix factorization to find patterns from similar users.")
    
    with col3:
        st.markdown("### 🧠 Neural Network")
        st.write("Combines 8 features to predict user preferences.")
    
    st.divider()
    
    st.subheader("Model Performance")
    perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
    perf_col1.metric("Hit Rate@20", "~70-80%")
    perf_col2.metric("Recall@20", "~40-50%")
    perf_col3.metric("Precision@5", "~15-18%")
    perf_col4.metric("NDCG@20", "~30-35%")
