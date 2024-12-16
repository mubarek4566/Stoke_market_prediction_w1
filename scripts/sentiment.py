import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import nltk
nltk.download('vader_lexicon')

import pandas as pd
from textblob import TextBlob

class SentimentAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        """Load the Excel file into a DataFrame."""
        self.data = pd.read_excel(self.file_path)
        if 'headline' not in self.data.columns:
            raise ValueError("The file must contain a 'headline' column.")

    def analyze_sentiment(self):
        """Perform sentiment analysis on the 'headline' column."""
        if self.data is None:
            raise ValueError("Data not loaded. Please run 'load_data' first.")

        def get_sentiment(text):
            """Classify the sentiment as Positive, Negative, or Neutral."""
            analysis = TextBlob(str(text))
            if analysis.sentiment.polarity > 0:
                return 'Positive'
            elif analysis.sentiment.polarity < 0:
                return 'Negative'
            else:
                return 'Neutral'

        # Apply sentiment analysis to the headline column
        self.data['Sentiment'] = self.data['headline'].apply(get_sentiment)

    def save_results(self, output_file):
        """Save the DataFrame with sentiment analysis results to a new Excel file."""
        if self.data is None or 'Sentiment' not in self.data.columns:
            raise ValueError("No sentiment analysis results to save. Please run 'analyze_sentiment' first.")
        
        self.data.to_excel(output_file, index=False)

# Usage Example
# Initialize the analyzer with the file path
file_path = '/mnt/data/sample.xlsx'
output_file = '/mnt/data/sentiment_results.xlsx'

analyzer = SentimentAnalyzer(file_path)
analyzer.load_data()
analyzer.analyze_sentiment()
analyzer.save_results(output_file)

print(f"Sentiment analysis completed. Results saved to {output_file}")
