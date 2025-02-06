import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import os
import logging
from typing import Tuple, Any
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from wordcloud import WordCloud

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='hotel_review_system.log'
)

class DataAnalyzer:
    """Class for performing exploratory data analysis"""
    
    @staticmethod
    def plot_rating_distribution(df):
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x='Rating')
        plt.title('Distribution of Hotel Ratings')
        plt.xlabel('Rating')
        plt.ylabel('Count')
        plt.savefig('visualizations/rating_distribution.png')
        plt.close()

    @staticmethod
    def create_wordcloud(df, sentiment):
        text = ' '.join(df[df['Sentiment'] == sentiment]['Review'])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud for {sentiment} Reviews')
        plt.savefig(f'visualizations/wordcloud_{sentiment.lower()}.png')
        plt.close()

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, labels):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig('visualizations/confusion_matrix.png')
        plt.close()

class ReviewPreprocessor:
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
        except Exception as e:
            logging.warning(f"NLTK download warning: {e}")
        
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess the text."""
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Tokenize
        tokens = word_tokenize(text)
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words]
        return ' '.join(tokens)

class HotelReviewSystem:
    def __init__(self, data_path: str):
        self.preprocessor = ReviewPreprocessor()
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.model = SVC(kernel='linear', probability=True)
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.data_path = data_path
        self.analyzer = DataAnalyzer()
        
        # Create directories for saving artifacts
        os.makedirs('models', exist_ok=True)
        os.makedirs('visualizations', exist_ok=True)

    def load_and_prepare_data(self) -> pd.DataFrame:
        """Load and prepare the dataset."""
        try:
            print("Loading dataset...")
            df = pd.read_csv(self.data_path)
            logging.info(f"Successfully loaded dataset from {self.data_path}")
            
            # Clean the data
            df['Review'] = df['Review'].fillna('')
            
            # Create sentiment labels based on rating
            df['Sentiment'] = pd.cut(
                df['Rating'], 
                bins=[0, 2, 3, 5], 
                labels=['Negative', 'Neutral', 'Positive']
            )
            
            # Generate visualizations
            print("Generating visualizations...")
            self.analyzer.plot_rating_distribution(df)
            for sentiment in ['Positive', 'Negative', 'Neutral']:
                self.analyzer.create_wordcloud(df, sentiment)
            
            return df
            
        except Exception as e:
            logging.error(f"Error loading dataset: {str(e)}")
            raise

    def train_model(self):
        """Train the sentiment analysis model."""
        try:
            print("Training model...")
            df = self.load_and_prepare_data()
            
            # Preprocess reviews
            X = df['Review'].apply(self.preprocessor.preprocess_text)
            y = df['Sentiment']
            
            # Encode labels
            y_encoded = self.label_encoder.fit_transform(y)
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42
            )
            
            # Vectorize text
            X_train_vec = self.vectorizer.fit_transform(X_train)
            X_test_vec = self.vectorizer.transform(X_test)
            
            # Train model
            self.model.fit(X_train_vec, y_train)
            
            # Perform cross-validation
            cv_scores = cross_val_score(self.model, X_train_vec, y_train, cv=5)
            print(f"\nCross-validation scores: {cv_scores}")
            print(f"Average CV score: {cv_scores.mean():.2f}")
            
            # Evaluate model
            y_pred = self.model.predict(X_test_vec)
            
            # Generate confusion matrix
            self.analyzer.plot_confusion_matrix(
                y_test, 
                y_pred, 
                self.label_encoder.classes_
            )
            
            # Print classification report
            print("\nModel Performance:")
            print(classification_report(y_test, y_pred, 
                                     target_names=self.label_encoder.classes_))
            
            self.is_trained = True
            self.save_model()
            
        except Exception as e:
            logging.error(f"Error training model: {str(e)}")
            raise

    def save_model(self):
        """Save the trained model and vectorizer."""
        try:
            joblib.dump(self.model, 'models/sentiment_model.pkl')
            joblib.dump(self.vectorizer, 'models/vectorizer.pkl')
            joblib.dump(self.label_encoder, 'models/label_encoder.pkl')
            logging.info("Model saved successfully")
        except Exception as e:
            logging.error(f"Error saving model: {str(e)}")
            raise

    def load_model(self):
        """Load the trained model and vectorizer."""
        try:
            self.model = joblib.load('models/sentiment_model.pkl')
            self.vectorizer = joblib.load('models/vectorizer.pkl')
            self.label_encoder = joblib.load('models/label_encoder.pkl')
            self.is_trained = True
            logging.info("Model loaded successfully")
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise

    def predict_sentiment(self, review: str) -> dict:
        """Predict sentiment for a given review."""
        try:
            if not self.is_trained:
                self.load_model()
            
            # Preprocess the review
            processed_review = self.preprocessor.preprocess_text(review)
            
            # Vectorize the review
            review_vec = self.vectorizer.transform([processed_review])
            
            # Get prediction and probabilities
            sentiment = self.label_encoder.inverse_transform(
                self.model.predict(review_vec)
            )[0]
            probabilities = self.model.predict_proba(review_vec)[0]
            
            # Create confidence scores
            confidence_scores = {
                label: float(prob) 
                for label, prob in zip(self.label_encoder.classes_, probabilities)
            }
            
            return {
                'sentiment': sentiment,
                'confidence_scores': confidence_scores,
                'processed_text': processed_review
            }
            
        except Exception as e:
            logging.error(f"Error predicting sentiment: {str(e)}")
            raise

def main():
    """Main function to run the hotel review system."""
    try:
        # Initialize the system with the correct path to your dataset
        data_path = 'tripadvisor_hotel_reviews.csv'  # Update this path to match your dataset location
        print("\nInitializing Hotel Review System...")
        review_system = HotelReviewSystem(data_path)
        
        while True:
            print("\n=== Hotel Review Sentiment Analysis System ===")
            print("1. Analyze a review")
            print("2. Retrain model")
            print("3. View visualizations")
            print("4. Exit")
            
            choice = input("\nEnter your choice (1-4): ")
            
            if choice == '1':
                review = input("\nEnter your hotel review: ")
                result = review_system.predict_sentiment(review)
                
                print("\n=== Analysis Results ===")
                print(f"Sentiment: {result['sentiment']}")
                print("\nConfidence Scores:")
                for sentiment, score in result['confidence_scores'].items():
                    print(f"{sentiment}: {score:.2%}")
                    
            elif choice == '2':
                print("\nRetraining model...")
                review_system.train_model()
                print("Model retrained successfully!")
                
            elif choice == '3':
                print("\nVisualizations have been saved in the 'visualizations' folder:")
                print("1. Rating distribution")
                print("2. Word clouds for each sentiment")
                print("3. Confusion matrix")
                
            elif choice == '4':
                print("\nThank you for using the Hotel Review System!")
                break
                
            else:
                print("\nInvalid choice. Please try again.")
                
    except Exception as e:
        logging.error(f"Application error: {str(e)}")
        print(f"An error occurred: {str(e)}")
        print("Please check the logs for details.")

if __name__ == "__main__":
    main()
