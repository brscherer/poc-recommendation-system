# POC Recommendation System

This project is a proof of concept for a recommendation system. It uses collaborative filtering to provide personalized recommendations.

## Algorithm

The recommendation system uses content-based filtering where we check director and key actors to calculate the cosine similarity and use the top 10 with higher scores to recommend movies to the user

## Setup

To run this application locally, follow these steps:

1. **Clone the repository:**
  ```bash
  git clone https://github.com/yourusername/poc-recommendation-system.git
  cd poc-recommendation-system
  ```

2. **Create a virtual environment:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

3. **Install the dependencies:**
  ```bash
  pip install -r requirements.txt
  ```

## Usage

To use the recommendation system, follow these steps:

1. **Run the application:**
  ```bash
  python app.py
  ```

2. Follow On-Screen Prompts:

