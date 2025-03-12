import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

#############################################
# Helper Functions to Load and Prepare Data #
#############################################

def load_movies_metadata():
    """Load movies metadata and perform basic cleaning."""
    movies = pd.read_csv('movies_metadata.csv', low_memory=False)
    # Some rows have non-numeric IDs.
    movies = movies[pd.to_numeric(movies['id'], errors='coerce').notnull()]
    movies['id'] = movies['id'].astype(int)
    movies['overview'] = movies['overview'].fillna('')
    movies.rename(columns={'title': 'movieName'}, inplace=True)
    return movies

def load_links():
    """Load links_small.csv to map MovieLens movieId to TMDB id."""
    links = pd.read_csv('links_small.csv')
    links = links[['movieId', 'tmdbId']]
    links['tmdbId'] = pd.to_numeric(links['tmdbId'], errors='coerce')
    links = links.dropna(subset=['tmdbId'])
    links['tmdbId'] = links['tmdbId'].astype(int)
    return links

def load_ratings():
    """Load ratings data and convert movieId to TMDB id using links_small.csv."""
    ratings = pd.read_csv('ratings_small.csv')
    links = load_links()
    # Merge ratings with links to convert movieId to tmdbId.
    ratings = ratings.merge(links, on='movieId', how='left')
    ratings = ratings.dropna(subset=['tmdbId'])
    ratings['tmdbId'] = ratings['tmdbId'].astype(int)
    ratings.rename(columns={'tmdbId': 'id'}, inplace=True)
    return ratings

def compute_movie_views(ratings):
    """Compute the number of ratings per movie (proxy for 'views')."""
    views = ratings.groupby('id')['rating'].count().reset_index()
    views.rename(columns={'rating': 'views'}, inplace=True)
    return views

############################################
# Functions to Process Credits Information #
############################################

def load_credits():
    """Load credits data (cast and crew information)."""
    credits = pd.read_csv('credits.csv') # Do not forget to import this file locally
    credits['id'] = pd.to_numeric(credits['id'], errors='coerce')
    credits = credits.dropna(subset=['id'])
    credits['id'] = credits['id'].astype(int)
    return credits

def extract_director(crew_str):
    """Extract director name from crew data."""
    try:
        crew = ast.literal_eval(crew_str)
    except Exception:
        return ''
    for member in crew:
        if member.get('job') == 'Director':
            return member.get('name', '')
    return ''

def extract_actors(cast_str):
    """Extract names of the top three actors from cast data."""
    try:
        cast = ast.literal_eval(cast_str)
    except Exception:
        return ''
    names = [member.get('name', '') for member in cast[:3]]
    return ' '.join(names)

def create_combined_features(movies, credits):
    """
    Merge movies metadata with credits data and create a combined feature string
    that includes the director and top actors.
    """
    movies = movies.merge(credits[['id', 'cast', 'crew']], on='id', how='left')
    movies['director'] = movies['crew'].apply(extract_director)
    movies['actors'] = movies['cast'].apply(extract_actors)
    movies['combined_features'] = movies['director'] + " " + movies['actors']
    return movies

def build_credit_matrix(movies):
    """
    Build a TF-IDF matrix using the combined features (director and actors).
    This matrix is used to compute similarity between movies.
    """
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['combined_features'].fillna(''))
    return tfidf_matrix

##################################
# Content-Based Filtering Method #
##################################

def get_content_recommendations_by_credits(movie_title, movies, tfidf_matrix, views_df, top_n=10):
    """
    Given a movie title, find top_n similar movies using cosine similarity on the
    combined director and actor features.
    
    Returns recommendations in the format:
      { movieName, views, imdbScore }
    """
    movies = movies.reset_index(drop=True)
    indices = pd.Series(movies.index, index=movies['movieName']).drop_duplicates()
    if movie_title not in indices:
        print(f"Movie '{movie_title}' not found in the database.")
        return []
    
    idx = indices[movie_title]
    cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    
    sim_scores = list(enumerate(cosine_sim))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]  # Exclude the input movie
    
    recs = []
    for i, score in sim_scores:
        movie_data = movies.iloc[i]
        movie_name = movie_data['movieName']
        imdbScore = movie_data.get('vote_average', np.nan)  # proxy for imdbScore
        movie_id = movie_data['id']
        view_val = views_df[views_df['id'] == movie_id]['views']
        views_count = int(view_val.iloc[0]) if not view_val.empty else 0
        recs.append({'movieName': movie_name, 'views': views_count, 'imdbScore': float(imdbScore)})
    return recs

###############
# Main Script #
###############

def main():
    print("Loading movies metadata...")
    movies = load_movies_metadata()
    print("Loading ratings data...")
    ratings = load_ratings()
    views_df = compute_movie_views(ratings)
    
    print("Loading credits data...")
    credits = load_credits()
    
    print("Merging movies with credits and building combined features...")
    movies = create_combined_features(movies, credits)
    
    print("Building TF-IDF matrix on combined director and actors data...")
    tfidf_matrix = build_credit_matrix(movies)
    
    print("Welcome to the Movie Recommendation System (Content-Based using Director & Actors)!")
    movie_title = input("Enter a movie title (e.g., 'The Godfather'): ").strip()
    
    recs = get_content_recommendations_by_credits(movie_title, movies, tfidf_matrix, views_df, top_n=10)
    if recs:
        print(f"\nMovies similar to '{movie_title}':")
        for r in recs:
            print(r)

if __name__ == '__main__':
    main()
