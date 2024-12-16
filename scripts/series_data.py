import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Data_visulization import EDA
class TimeSeries:
    def __init__(self, dataframe):
        """
        Initializes the TimeSeries class with the provided DataFrame and preprocesses the date column.
        """
        eda = EDA(dataframe)
        eda.parse_dates()
        self.dataframe = eda.dataframe  # Preprocess the DataFrame during initialization

    '''
    Let's start by examining the content of your uploaded Excel file. I’ll read it to identify the structure and determine how to apply TA-Lib for calculating technical indicators.

    The dataset contains the following columns:

    Date: Dates of stock prices.
    Open, High, Low, Close: Daily stock price metrics.
    Adj Close: Adjusted closing prices.
    Volume: Number of shares traded.
    Dividends: Dividend amounts (all zeros in this sample).
    Stock Splits: Split details (also all zeros in this sample).
    I will now demonstrate how to use TA-Lib to calculate some technical indicators like:

    Moving Averages (SMA and EMA).
    RSI (Relative Strength Index).
    MACD (Moving Average Convergence Divergence). '''

    def stock_Analysis(self):
        
    




    def analyze_publishers(self):
        """
        Analyze the publishers contributing to the news feed. Identify which publishers are most active,
        and check if there are unique domains or organizations contributing more frequently. 
        The analysis will also consider sentiment and type of news reported.
        """
        if 'publisher' not in self.dataframe.columns:
            raise ValueError("The dataframe does not contain a 'publisher' column. Please provide the correct input.")

        # Count the number of articles published by each publisher
        publisher_counts = self.dataframe['publisher'].value_counts()

        # Plot the number of articles published by each publisher
        plt.figure(figsize=(12, 6))
        publisher_counts.head(10).plot(kind='bar', color='lightblue')
        plt.title("Top 10 Most Active Publishers")
        plt.xlabel("Publisher")
        plt.ylabel("Number of Articles Published")
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.show()

        # If email addresses are used as publisher names, extract unique domains
        if self.dataframe['publisher'].str.contains('@').any():
            self.dataframe['publisher_domain'] = self.dataframe['publisher'].apply(
                lambda x: x.split('@')[-1] if isinstance(x, str) else None
            )
            domain_counts = self.dataframe['publisher_domain'].value_counts()

            # Plot the most frequent publisher domains
            plt.figure(figsize=(12, 6))
            domain_counts.head(10).plot(kind='bar', color='orange')
            plt.title("Top 10 Most Frequent Publisher Domains")
            plt.xlabel("Domain")
            plt.ylabel("Number of Articles Published")
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.show()

        # Analyzing sentiment per publisher to see which publishers have positive, negative, or neutral news
        if 'sentiment' in self.dataframe.columns:
            publisher_sentiment = self.dataframe.groupby('publisher')['sentiment'].mean()

            # Plot the average sentiment of articles by publisher
            plt.figure(figsize=(12, 6))
            publisher_sentiment.sort_values().head(10).plot(kind='bar', color='purple')
            plt.title("Average Sentiment of Articles by Publisher")
            plt.xlabel("Publisher")
            plt.ylabel("Average Sentiment")
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.show()


    def analyze_correlation(self, stock_df):
        """
        Analyze the correlation between sentiment scores and stock price changes.
        """
        self.dataframe['date'] = self.dataframe['date'].dt.date
        stock_df['Date'] = pd.to_datetime(stock_df['Date'])

        stock_df['Price_Change'] = stock_df['Close'].pct_change()
        sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
        self.dataframe['sentiment'] = self.dataframe['headline'].apply(
            lambda x: sentiment_map.get(x, 0)
        )

        merged_df = pd.merge(
            self.dataframe, stock_df,
            left_on=['date', 'stock'],
            right_on=['Date', 'company'],
            how='inner'
        )

        correlation_data = merged_df.groupby('sentiment')['Price_Change'].mean()
        print("Correlation data:")
        print(correlation_data)

        corr_matrix = merged_df[['sentiment', 'Price_Change']].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        plt.title("Correlation Between Sentiment and Stock Price Change")
        plt.show()