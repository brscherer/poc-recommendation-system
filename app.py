import pandas as pd
from surprise import Dataset, Reader, SVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# import warnings

# warnings.filterwarnings("ignore")

#############################################
# Helper Functions to Load and Prepare Data #
#############################################


def load_movies_metadata():
    """Load movies metadata and perform basic cleaning."""
    movies = pd.read_csv("movies_metadata.csv", low_memory=False)
    # Some rows in movies_metadata.csv have non-numeric IDs.
    movies = movies[pd.to_numeric(movies["id"], errors="coerce").notnull()]
    movies["id"] = movies["id"].astype(int)

    # Replace NaN in important text fields with empty string
    movies["overview"] = movies["overview"].fillna("")

    # For this demo, we assume 'vote_average' corresponds to the imdbScore proxy.
    # We also add a temporary 'movieName' column from 'title'
    movies.rename(columns={"title": "movieName"}, inplace=True)

    return movies


def load_ratings():
    """Load ratings data from ratings_small.csv."""
    ratings = pd.read_csv("ratings_small.csv")
    # Ensure movieId is numeric and rename column to 'id' for joining with movies_metadata
    ratings["movieId"] = pd.to_numeric(ratings["movieId"], errors="coerce")
    ratings = ratings.dropna(subset=["movieId"])
    ratings["movieId"] = ratings["movieId"].astype(int)
    ratings.rename(columns={"movieId": "id"}, inplace=True)
    return ratings


def compute_movie_views(ratings):
    """Compute number of ratings per movie (used as a proxy for 'views')."""
    views = ratings.groupby("id")["rating"].count().reset_index()
    views.rename(columns={"rating": "views"}, inplace=True)
    return views


##############################
# 1. Collaborative Filtering #
##############################


def build_cf_model(ratings):
    """
    Build a collaborative filtering model using Surprise's SVD.
    The model is trained on the ratings_small.csv file.
    """
    # Define a reader with rating scale (MovieLens ratings are 1-5)
    reader = Reader(rating_scale=(ratings["rating"].min(), ratings["rating"].max()))

    # Surprise expects a dataframe with columns: userID, itemID, rating
    data = Dataset.load_from_df(ratings[["userId", "id", "rating"]], reader)

    # Split for training (here we use all data for training for simplicity)
    trainset = data.build_full_trainset()

    # Use SVD as the matrix factorization algorithm
    model = SVD()
    model.fit(trainset)

    return model


def get_cf_recommendations(user_id, model, movies, ratings, views_df, top_n=10):
    """
    Get top_n movie recommendations for a given user_id using the CF model.
    Returns a list of dictionaries with keys: movieName, views, imdbScore.
    """
    # Get a list of all movie ids and titles
    movie_ids = movies["id"].unique()

    # Get movies that the user has already rated
    user_rated = ratings[ratings["userId"] == user_id]["id"].unique()

    predictions = []
    for mid in movie_ids:
        if mid in user_rated:
            continue
        est_rating = model.predict(user_id, mid).est
        predictions.append((mid, est_rating))

    # Sort by estimated rating in descending order and take top_n
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_predictions = predictions[:top_n]

    # Prepare final recommendations
    recs = []
    for mid, score in top_predictions:
        # Get movie metadata
        movie_data = movies[movies["id"] == mid].iloc[0]
        movie_name = movie_data["movieName"]
        imdbScore = movie_data["vote_average"]  # proxy for imdbScore
        # Get views (if missing, set as 0)
        view_val = views_df[views_df["id"] == mid]["views"]
        views_count = int(view_val.iloc[0]) if not view_val.empty else 0
        recs.append(
            {"movieName": movie_name, "views": views_count, "imdbScore": imdbScore}
        )
    return recs


##################################
# 2. Content-Based Filtering     #
##################################


def build_content_matrix(movies):
    """
    Build a TF-IDF matrix on the 'overview' column.
    """
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies["overview"])
    return tfidf_matrix


def get_content_recommendations(movie_title, movies, tfidf_matrix, views_df, top_n=10):
    """
    Given a movie title, find top_n similar movies using cosine similarity on overviews.
    Returns recommendations in the specified output format.
    """
    # Reset index so we can use indices easily
    movies = movies.reset_index(drop=True)

    # Check if the movie exists
    indices = pd.Series(movies.index, index=movies["movieName"]).drop_duplicates()
    if movie_title not in indices:
        print(f"Movie '{movie_title}' not found in the database.")
        return []

    idx = indices[movie_title]

    # Compute cosine similarity between the chosen movie and all others
    cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

    # Get indices of the top_n+1 movies (the first one is the movie itself)
    sim_scores = list(enumerate(cosine_sim))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1 : top_n + 1]  # exclude the movie itself

    recs = []
    for i, score in sim_scores:
        movie_data = movies.iloc[i]
        movie_name = movie_data["movieName"]
        imdbScore = movie_data["vote_average"]
        movie_id = movie_data["id"]
        view_val = views_df[views_df["id"] == movie_id]["views"]
        views_count = int(view_val.iloc[0]) if not view_val.empty else 0
        recs.append(
            {"movieName": movie_name, "views": views_count, "imdbScore": imdbScore}
        )
    return recs


##########################################
# 3. Hybrid Recommendation System        #
##########################################


def get_hybrid_recommendations(
    user_id,
    favorite_movie,
    model,
    movies,
    ratings,
    views_df,
    tfidf_matrix,
    alpha=0.5,
    top_n=10,
):
    """
    Hybrid recommendation that combines collaborative filtering score with content similarity.

    Parameters:
      - user_id: The user id for CF predictions.
      - favorite_movie: A movie title representing the user's current interest.
      - alpha: Weight for collaborative filtering (0.0 to 1.0); (1 - alpha) is weight for content.

    The final score is computed as:
      final_score = alpha * CF_score_normalized + (1 - alpha) * content_similarity
    """
    # First, compute CF scores for all movies not rated by the user
    movie_ids = movies["id"].unique()
    user_rated = ratings[ratings["userId"] == user_id]["id"].unique()

    cf_scores = {}
    for mid in movie_ids:
        if mid in user_rated:
            continue
        pred = model.predict(user_id, mid).est
        cf_scores[mid] = pred
    # Normalize CF scores to range [0,1]
    if cf_scores:
        min_cf = min(cf_scores.values())
        max_cf = max(cf_scores.values())
        for mid in cf_scores:
            if max_cf - min_cf > 0:
                cf_scores[mid] = (cf_scores[mid] - min_cf) / (max_cf - min_cf)
            else:
                cf_scores[mid] = 0.0

    # Compute content similarity between favorite_movie and every movie in movies using the tfidf_matrix
    movies = movies.reset_index(drop=True)
    indices = pd.Series(movies.index, index=movies["movieName"]).drop_duplicates()
    if favorite_movie not in indices:
        print(
            f"Favorite movie '{favorite_movie}' not found. Hybrid recommendation will use only CF scores."
        )
        # Fall back to pure CF recommendations.
        recs = get_cf_recommendations(
            user_id, model, movies, ratings, views_df, top_n=top_n
        )
        return recs

    fav_idx = indices[favorite_movie]
    cosine_sim = cosine_similarity(tfidf_matrix[fav_idx], tfidf_matrix).flatten()
    # Create hybrid score for each movie not rated by the user.
    hybrid_scores = {}
    for idx, row in movies.iterrows():
        mid = row["id"]
        if mid in user_rated:
            continue
        cf_score = cf_scores.get(mid, 0.0)
        content_score = cosine_sim[idx]
        # Combine using weighted average
        hybrid_scores[mid] = alpha * cf_score + (1 - alpha) * content_score

    # Sort by hybrid score and return top_n movies.
    hybrid_sorted = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[
        :top_n
    ]
    recs = []
    for mid, score in hybrid_sorted:
        movie_data = movies[movies["id"] == mid].iloc[0]
        movie_name = movie_data["movieName"]
        imdbScore = movie_data["vote_average"]
        view_val = views_df[views_df["id"] == mid]["views"]
        views_count = int(view_val.iloc[0]) if not view_val.empty else 0
        recs.append(
            {"movieName": movie_name, "views": views_count, "imdbScore": imdbScore}
        )
    return recs


##########################
# Main Interactive Script#
##########################


def main():
    print("Loading data...")
    movies = load_movies_metadata()
    ratings = load_ratings()
    views_df = compute_movie_views(ratings)

    print("Building models. This might take a moment...")
    # Collaborative Filtering Model
    cf_model = build_cf_model(ratings)
    # Content-Based Matrix
    tfidf_matrix = build_content_matrix(movies)

    print("Welcome to the Movie Recommendation System!")
    print("Choose an option:")
    print("1 - Collaborative Filtering (Enter your user id)")
    print("2 - Content-Based Filtering (Enter a movie title)")
    print("3 - Hybrid Recommendation (Enter your user id and a favorite movie)")

    choice = input("Enter option (1, 2, or 3): ").strip()

    if choice == "1":
        try:
            user_id = int(input("Enter your user id (integer): ").strip())
        except ValueError:
            print("Invalid user id. Please enter an integer.")
            return
        recs = get_cf_recommendations(
            user_id, cf_model, movies, ratings, views_df, top_n=10
        )
        print(f"\nTop recommendations for user {user_id} (Collaborative Filtering):")
        for r in recs:
            print(r)

    elif choice == "2":
        movie_title = input("Enter a movie title (e.g., 'The Godfather'): ").strip()
        recs = get_content_recommendations(
            movie_title, movies, tfidf_matrix, views_df, top_n=10
        )
        if recs:
            print(f"\nMovies similar to '{movie_title}' (Content-Based):")
            for r in recs:
                print(r)

    elif choice == "3":
        try:
            user_id = int(input("Enter your user id (integer): ").strip())
        except ValueError:
            print("Invalid user id. Please enter an integer.")
            return
        favorite_movie = input(
            "Enter a favorite movie title to guide recommendations: "
        ).strip()
        recs = get_hybrid_recommendations(
            user_id,
            favorite_movie,
            cf_model,
            movies,
            ratings,
            views_df,
            tfidf_matrix,
            alpha=0.5,
            top_n=10,
        )
        print(
            f"\nHybrid Recommendations for user {user_id} (based on CF and similarity to '{favorite_movie}'):"
        )
        for r in recs:
            print(r)
    else:
        print("Invalid option. Please run the script again and choose 1, 2, or 3.")


if __name__ == "__main__":
    main()
