import pandas as pd
from pyti.simple_moving_average import simple_moving_average as sma
from pyti.relative_strength_index import relative_strength_index as rsi
from pyti.moving_average_convergence_divergence import moving_average_convergence_divergence as macd
from pyti.exponential_moving_average import exponential_moving_average as ema

#from numpy import NaN as npNaN
#import pandas_ta as ta
import pynance as pn
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from scipy.stats import spearmanr

class FinancialAnalysis:
    def __init__(self, dataframe):
        self.df = dataframe
    
    def calculate_indicators(self, period=14):
        # Ensure data is sorted
        self.df = self.df.sort_values(by="Date")

        # SMA and RSI Calculation
        self.df["SMA_20"] = sma(self.df["Close"].tolist(), period=50)
        self.df["RSI"] = rsi(self.df["Close"].tolist(), period=period)

        # MACD Calculation
        macd_values = macd(self.df["Close"].tolist(), short_period=12, long_period=26)
        self.df["MACD"] = macd_values  # MACD line (directly returned by pyti's macd())

        # Signal Line Calculation (SMA of MACD values)
        signal_line = sma(macd_values, period=9)
        self.df["MACD_signal"] = signal_line

        # Histogram Calculation (MACD - Signal Line)
        self.df["MACD_hist"] = self.df["MACD"] - self.df["MACD_signal"]

        self.df["Close"] = pd.to_numeric(self.df["Close"], errors='coerce')
        self.df.dropna(subset=["Close"], inplace=True)

    def visualize_SMA(self):
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.df["Date"], self.df["Close"], label="Close Price", color="blue")
        plt.plot(self.df["Date"], self.df["SMA_20"], label="SMA_20", color="orange")
        plt.legend()
        plt.show()

    def visualize_RSI(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.df["Date"], self.df["RSI"], label="RSI", color="green")
        plt.axhline(70, linestyle="--", color="red", label="Overbought")
        plt.axhline(30, linestyle="--", color="blue", label="Oversold")
        plt.legend()
        plt.show()

    def visualize_MACD(self):
        #Plot MACD, signal line, and histogram to identify bullish/bearish momentum.
        plt.figure(figsize=(12, 6))
        plt.plot(self.df["Date"], self.df["MACD"], label="MACD", color="purple")
        plt.plot(self.df["Date"], self.df["MACD_signal"], label="Signal Line", color="orange")
        plt.bar(self.df["Date"], self.df["MACD_hist"], label="Histogram", color="gray")
        plt.legend()
        plt.show()

    def FinancialMetrics(self, symbol='AAPL', start='2020-01-01', end='2024-12-15'):
        """
        Fetch financial data for a given stock symbol and date range.
        """
        data = pn.data.get(symbol, start=start, end=end)
        return data
    
    def Correlation_news_stock(self):
        sentiment_map = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
        self.df['sentiment_score'] = self.df['sentiment'].map(sentiment_map)

        # Aggregate sentiment scores for each day
        daily_sentiment = self.df.groupby('date')['sentiment_score'].mean().reset_index()
        daily_data = pd.merge(daily_sentiment, self.df[['date', 'daily_return']].drop_duplicates(), on='date')

        # Drop rows with missing data
        daily_data = daily_data.dropna()

        # Compute Pearson correlation
        correlation, p_value = pearsonr(daily_data['sentiment_score'], daily_data['daily_return'])
        print(f"Pearson Correlation: {correlation:.4f} (p-value: {p_value:.4e})")

        # Scatterplot
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=daily_data, x='sentiment_score', y='daily_return', color="blue")
        plt.title(f"Correlation Between Sentiment Scores and Stock Returns\n(Pearson r: {correlation:.2f})")
        plt.xlabel("Average Daily Sentiment Score")
        plt.ylabel("Daily Stock Return (%)")
        plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)
        plt.show()
        
    def Spearman_Corr_news_stock(self):
        required_columns = ['sentiment', 'daily_return']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        
        if missing_columns:
            raise KeyError(f"The following required columns are missing: {missing_columns}")
        
        # Map sentiments to numerical scores (if not already done)
        sentiment_map = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
        if 'sentiment_score' not in self.df.columns:
            self.df['sentiment_score'] = self.df['sentiment'].map(sentiment_map)
        
        # Drop rows with NaN in the necessary columns
        self.df.dropna(subset=['sentiment_score', 'daily_return'], inplace=True)
        
        # Compute Spearman correlation
        correlation, p_value = spearmanr(self.df['sentiment_score'], self.df['daily_return'])
        print(f"Spearman Correlation: {correlation:.4f} (p-value: {p_value:.4e})")
        
        # Scatterplot
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=self.df, x='sentiment_score', y='daily_return', color="blue")
        plt.title(f"Correlation Between Sentiment Scores and Stock Returns\n(Spearman ρ: {correlation:.2f})")
        plt.xlabel("Sentiment Score")
        plt.ylabel("Daily Stock Return (%)")
        plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)
        plt.show()

        