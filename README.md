# Hotel Review Sentiment Analysis

Classifies hotel reviews as positive, neutral, or negative using machine learning.

## Overview

This project uses an SVM model with TF-IDF to analyze sentiment in hotel reviews from TripAdvisor.

## Usage

1.  **Clone:** `git clone https://github.com/srishtiYadav06/Hotel_Review_Sentiment_Analysis`
2.  **Install:** `pip install scikit-learn nltk pandas matplotlib seaborn wordcloud`
3.  **Run:** `python main.py` (Ensure `hotel_reviews.csv` exists in the same directory)

## Data

*   `hotel_reviews.csv`: Contains reviews and ratings.

    *   Ratings 0-2: Negative
    *   Rating 3: Neutral
    *   Ratings 4-5: Positive

## Key Results

*   86% overall accuracy.
*   See full results in the main script's output.

## Future Improvements

*   Explore deep learning models.
*   Improve neutral sentiment classification.
