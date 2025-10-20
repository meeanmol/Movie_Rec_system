import streamlit as st
import pandas as pd
import numpy as np
import pickle
import difflib
import joblib

# Page configuration
st.set_page_config(
    page_title="CineMatch - Movie Recommendation Engine",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS with natural colors
st.markdown("""
<style>
    /* Main background with natural gradient */
    .stApp {
        background: #E8F5E9;
    }
    
    .main-header {
    font-size: 6rem;
    background: linear-gradient(45deg, #FF6B6B, #EE5A24);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 2rem;
    font-weight: 900;
    padding: 2rem 0;
    letter-spacing: 2px;
    text-shadow: 3px 3px 6px rgba(0,0,0,0.2);
}
    .sub-header {
        font-size: 1.5rem;
        color: #2D3748;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    
    .movie-card {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
        border: none;
        transition: transform 0.3s ease;
    }
    
    .movie-card:hover {
        transform: translateY(-5px);
    }
    
    .movie-card h4 {
        color: white;
        margin-bottom: 0.8rem;
        font-size: 1.3rem;
        font-weight: 600;
    }
    
    .similarity-badge {
        background: rgba(255, 255, 255, 0.3);
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        display: inline-block;
        backdrop-filter: blur(10px);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #a29bfe 0%, #6c5ce7 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
    }
    
   
    
    .stButton button {
        background: linear-gradient(45deg, #FF6B6B, #EE5A24);
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        border-radius: 50px;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(255, 107, 107, 0.4);
    }
    
    .stSelectbox, .stTextInput, .stSlider {
        background: white;
        border-radius: 10px;
        padding: 0.5rem;
    }
    
    /* Hide sidebar */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

class MovieRecommender:
    def __init__(self):
        self.model_path = "model_compressed.pkl"  
        self.data_path = "movies.csv"       
        self.model_components = None
        self.data = None
        self.similarity_matrix = None
        self.load_model()
    
    def load_model(self):
        """Load the pre-trained model components (vectorizer, similarity matrix, and data)"""
        try:
            # Load the model bundle from pickle file
            model_bundle = joblib.load("model_compressed.pkl")

            # Extract model components
            self.vectorizer = model_bundle.get('vectorizer', None)
            self.similarity_matrix = model_bundle.get('similarity_matrix', None)
            self.data = model_bundle.get('data', None)

            # Validation
            if self.data is not None:
                st.success(f"🎬 Loaded {len(self.data)} movies successfully!")
            else:
                st.warning("⚠️ Data not found inside the model file.")

            if self.vectorizer is None or self.similarity_matrix is None:
                st.warning("⚠️ Some components (vectorizer/similarity) are missing in model file.")
            else:
                st.info("✅ Model components loaded successfully!")

        except FileNotFoundError:
            st.error("❌ Model file not found. Please make sure the .pkl file exists in your project folder.")
            st.stop()
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            st.stop()

    
    def get_recommendations(self, movie_name, n_recommendations=10):
        """Get movie recommendations using the pre-trained model"""
        try:
            # Create list of all movie titles
            list_of_all_titles = self.data['title'].tolist()
            
            # Find close match for the movie name
            find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles, n=5, cutoff=0.1)
            
            if not find_close_match:
                return None, f"🎯 No close match found for '{movie_name}'"
            
            close_match = find_close_match[0]
            
            # Find the index of the movie
            movie_indices = self.data[self.data['title'] == close_match].index
            if len(movie_indices) == 0:
                return None, f"🎯 Movie '{close_match}' not found in dataset"
            
            index_of_the_movie = movie_indices[0]
            
            # Get similarity scores
            similarity_scores = list(enumerate(self.similarity_matrix[index_of_the_movie]))
            
            # Sort similar movies
            sorted_similar_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
            
            # Get top recommendations
            recommendations = []
            i = 1
            for movie in sorted_similar_movies:
                index = movie[0]
                title_from_index = self.data.iloc[index]['title']
                
                # Skip the input movie itself
                if title_from_index == close_match:
                    continue
                
                if i <= n_recommendations:
                    movie_data = self.data.iloc[index]
                    recommendations.append({
                        'title': title_from_index,
                        'genres': movie_data.get('genres', 'N/A'),
                        'overview': movie_data.get('overview', 'No overview available'),
                        'vote_average': movie_data.get('vote_average', 'N/A'),
                        'similarity_score': movie[1],
                        'index': index
                    })
                    i += 1
            
            return recommendations, close_match
            
        except Exception as e:
            return None, f"❌ Error getting recommendations: {str(e)}"

def display_recommendations_results(recommendations, matched_movie, show_similarity, show_details):
    """Display recommendations in a beautiful format"""
    st.markdown(f"""
    <div style='text-align: center; padding: 1rem; background: linear-gradient(135deg, #a29bfe 0%, #6c5ce7 100%); border-radius: 15px; margin: 2rem 0;'>
        <h2 style='color: white; margin: 0;'>🎉 Recommended movies similar to</h2>
        <h1 style='color: white; margin: 0; font-size: 2.5rem;'>"{matched_movie}"</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Display metrics
    avg_similarity = np.mean([movie['similarity_score'] for movie in recommendations])
    avg_rating_values = [movie['vote_average'] for movie in recommendations if movie['vote_average'] != 'N/A']
    if avg_rating_values:
        avg_rating = np.mean(avg_rating_values)
    else:
        avg_rating = 'N/A'
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <h3>📊 Recommendations</h3>
            <h1 style='font-size: 3rem; margin: 0;'>{len(recommendations)}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <h3>🎯 Avg Match</h3>
            <h1 style='font-size: 3rem; margin: 0;'>{avg_similarity*100:.1f}%</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        if avg_rating != 'N/A':
            st.markdown(f"""
            <div class='metric-card'>
                <h3>⭐ Avg Rating</h3>
                <h1 style='font-size: 3rem; margin: 0;'>{avg_rating:.1f}/10</h1>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='metric-card'>
                <h3>⭐ Avg Rating</h3>
                <h1 style='font-size: 3rem; margin: 0;'>N/A</h1>
            </div>
            """, unsafe_allow_html=True)
    
    # Display recommendations in grid
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: #2D3748;'>🎭 Your Personalized Recommendations</h2>", unsafe_allow_html=True)
    
    cols = st.columns(2)
    for idx, movie in enumerate(recommendations):
        with cols[idx % 2]:
            with st.container():
                # Movie card
                st.markdown(f"""
                <div class='movie-card'>
                    <h4>#{idx+1} {movie['title']}</h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Movie details
                if show_details:
                    st.write(f"**🎭 Genres:** {movie['genres']}")
                    
                    if movie['overview'] and movie['overview'] != 'No overview available':
                        st.write(f"**📖 Overview:** {movie['overview'][:150]}...")
                
                detail_col1, detail_col2 = st.columns(2)
                with detail_col1:
                    if movie['vote_average'] != 'N/A':
                        st.write(f"**⭐ Rating:** {movie['vote_average']}/10")
                
                with detail_col2:
                    if show_similarity:
                        similarity_percent = movie['similarity_score'] * 100
                        st.markdown(f"<div class='similarity-badge'>🎯 {similarity_percent:.1f}% match</div>", unsafe_allow_html=True)
                
                st.markdown("---")

def main():
    # Header
    st.markdown('<h1 class="main-header">CineMatch</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Discover Your Next Favorite Movie with ML Recommendations Model</p>', unsafe_allow_html=True)
    
    # Initialize recommender
    if 'recommender' not in st.session_state:
        with st.spinner("🚀 Loading movie database and AI model..."):
            st.session_state.recommender = MovieRecommender()
    
    recommender = st.session_state.recommender
    
    # Input section
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Movie selection
        movie_titles = recommender.data['title'].tolist()
        
        selected_movie = st.selectbox(
            "**🎬 Select a movie you love:**",
            movie_titles,
            index=0,
            help="Choose from our extensive movie collection"
        )
        
        # Alternative text input
        custom_movie = st.text_input(
            "**🔍 Or search for a movie:**",
            placeholder="Type any movie name...",
            help="We'll find the closest match in our database"
        )
        
        if custom_movie:
            selected_movie = custom_movie
        
        # Recommendation settings
        col3, col4 = st.columns(2)
        with col3:
            n_recommendations = st.slider(
                "**📈 Number of recommendations:**",
                min_value=5,
                max_value=20,
                value=10
            )
        
        with col4:
            show_similarity = st.checkbox("**🎯 Show match scores**", value=True)
            show_details = st.checkbox("**📖 Show movie details**", value=True)
    
    with col2:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #fd79a8 0%, #fdcb6e 100%); padding: 1.5rem; border-radius: 15px; color: white; height: 100%;'>
            <h3 style='color: white; margin-top: 0;'>✨ How It Works</h3>
            <p style='color: white; font-size: 0.9rem;'>
            • Tell us a movie you enjoy<br>
            • Our AI analyzes its features<br>
            • Discover similar movies instantly<br>
            • Find your next favorite!<br>
            • This model trained with Hollywood Movies!        
            </p>
            <div style='text-align: center; margin-top: 1rem;'>
                <h2 style='color: white; font-size: 2rem; margin: 0;'>{len(recommender.data)}</h2>
                <p style='color: white; margin: 0;'>Movies Ready</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close input section
    
    # Get recommendations button
    if st.button("🎬 Find Similar Movies", use_container_width=True):
        if selected_movie:
            with st.spinner("🔍 Analyzing movies and finding perfect matches..."):
                recommendations, matched_movie = recommender.get_recommendations(
                    selected_movie, n_recommendations
                )
                
                if recommendations:
                    display_recommendations_results(recommendations, matched_movie, show_similarity, show_details)
                else:
                    st.error(f"❌ {matched_movie}")
        else:
            st.warning("⚠️ Please select or enter a movie name first.")
    
    # Footer
    st.markdown("""
    <div style='text-align: center; margin-top: 3rem; padding: 2rem; background: rgba(255, 255, 255, 0.1); border-radius: 15px;'>
        <h2 style='color: #666; margin: 0; font-size: 1.1rem;'>
        Anmol Yadav | Powered by Machine Learning & Data Science
        </h2>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
