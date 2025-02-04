# POC - Recommendation System

This project is a Proof of Concept (POC) for a Recommendation System. The goal of this system is to provide personalized recommendations to users based on their preferences and behavior using [The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset?resource=download). 

## Features

- **User Profiling**: Collects and analyzes user data to create detailed user profiles.
- **Content-Based Filtering**: Recommends items similar to those the user has shown interest in.
- **Collaborative Filtering**: Suggests items based on the preferences of similar users.
- **Hybrid Approach**: Combines content-based and collaborative filtering for more accurate recommendations.
- **Real-Time Recommendations**: Provides up-to-date suggestions as user data changes.

## Technologies Used

- **Python**: The core programming language used for developing the recommendation algorithms.
- **Pandas**: For data manipulation and analysis.
- **Scikit-Learn**: For implementing machine learning models.
- **Flask**: To create a simple web interface for interacting with the recommendation system.
- **SQLite**: A lightweight database to store user data and preferences.

## Installation

1. Clone the repository:
  ```bash
  git clone https://github.com/yourusername/poc-recommendation-system.git
  ```
2. Navigate to the project directory:
  ```bash
  cd poc-recommendation-system
  ```
3. Install the required dependencies:
  ```bash
  pip install -r requirements.txt
  ```

## Usage

1. Run the app:
  ```bash
  python app.py
  ```
2. Follow On-Screen Prompts:
- Option 1 (Collaborative Filtering): Enter your user id (an integer as found in the ratings file) to get personalized recommendations.
- Option 2 (Content-Based Filtering): Enter a movie title (e.g., "The Godfather") to get similar movies.
- Option 3 (Hybrid Recommendation): Enter both your user id and a favorite movie title. The system will combine user preferences and content similarity for recommendations.
