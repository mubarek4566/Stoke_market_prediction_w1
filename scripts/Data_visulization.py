# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Set visualization style
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

class EDA:
    def __init__(self, dataframe):
        """
        Initializes the SentimentEDA class with the merged dataframe.
        """
        self.dataframe = dataframe

    def parse_dates(self):
        #Parses and converts dates to UTC format in the 'date' column. 

        def parse_date(date):
            try:       
                dt = pd.to_datetime(date, format='%Y-%m-%d %H:%M:%S')
            except:
                try:
                    # Attempt automatic inference if the first attempt fails
                    dt = pd.to_datetime(date, errors='coerce')
                except:
                    return pd.NaT  # If all parsing fails, return NaT
            # Ensure tz-aware (convert to UTC)
            return dt.tz_convert('UTC') if dt is not pd.NaT and dt.tzinfo else dt.tz_localize('UTC')

        if 'date' not in self.dataframe.columns:
            raise ValueError("The dataframe does not contain a 'date' column.")
        
        # Apply parsing to the 'date' column
        self.dataframe['date'] = self.dataframe['date'].apply(parse_date)
        # Drop rows with invalid dates (NaT)
        self.dataframe = self.dataframe.dropna(subset=['date'])
        # Sort the DataFrame by 'date'
        self.dataframe = self.dataframe.sort_values(by='date')

        # Basic Information and Structure
    def display_info(self):
        print("Dataset Info:")
        print(self.dataframe.info())
        print("\nMissing Values:")
        print(self.dataframe.isnull().sum())
        print(f"\nDataset Shape: {self.dataframe.shape}")

    def stat_summary(self):
        print("\nStatistical Summary (Numerical Features):")
        print(self.dataframe.describe())
        print("\nStatistical Summary (Categorical Features):")
        print(self.dataframe.describe(include='object'))

    def duplicates(self):
        # Step 1: Check for duplicate rows
        print("Checking for Duplicate Rows...")
        duplicates = self.dataframe.duplicated()
        if not duplicates.empty:
            print("\nDuplicate Rows Found:")
            print(duplicates)
        else:
            print("No duplicate rows found.")
    
    def univariate_num(self):
        """
        Univariate Analysis for Numerical Columns Distribution.
        """
        # Select only numerical columns
        numerical_cols = self.dataframe.select_dtypes(include=['int64', 'float64']).columns
        
        # Define grid size: Calculate rows and columns dynamically
        num_cols = 3  # Number of columns in the grid
        num_plots = len(numerical_cols)  # Total number of plots
        num_rows = math.ceil(num_plots / num_cols)  # Calculate required rows dynamically
        
        # Create a figure with subplots
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
        axes = axes.flatten()  # Flatten axes for easier iteration
        
        # Loop through numerical columns and plot histograms
        for idx, col in enumerate(numerical_cols):
            sns.histplot(self.dataframe[col], kde=True, bins=30, ax=axes[idx])  # Histogram with KDE
            axes[idx].set_title(f"Distribution of {col}")  # Set title
            axes[idx].set_xlabel(col)  # Label x-axis
            axes[idx].set_ylabel("Frequency")  # Label y-axis
        
        # Hide any unused subplots
        for i in range(len(numerical_cols), len(axes)):
            fig.delaxes(axes[i])
        
        # Adjust layout to prevent overlapping
        plt.tight_layout()
        plt.show()


    def bivariate_num(self):
        # Bivariate Analysis for Numerical vs Numerical (Scatterplots)
        numerical_cols = self.dataframe.select_dtypes(include=['int64', 'float64']).columns
        num_plots = len(numerical_cols) - 1  # Exclude the first column as x-axis
        num_cols = 3  # Number of columns in the grid
        num_rows = math.ceil(num_plots / num_cols)  # Calculate rows needed for grid
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))  # Flexible grid size
        axes = axes.flatten()  # Flatten in case of multi-dimensional axes
        
        x_col = numerical_cols[0]  # Use the first numerical column as x-axis
        plot_idx = 0  # Track plot index

        for col in numerical_cols:
            if col != x_col:  # Skip the x-axis column itself
                ax = axes[plot_idx]
                sns.scatterplot(data=self.dataframe, x=x_col, y=col, ax=ax)
                ax.set_title(f"{x_col} vs {col}")
                ax.set_xlabel(x_col)
                ax.set_ylabel(col)
                plot_idx += 1
        
        # Hide any unused subplots
        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()

    def correlation_analysis(self):
        # Plots a heatmap of correlations for numerical features in the dataset.
        numerical_data = self.dataframe.select_dtypes(include=['float64', 'int64'])
        if numerical_data.empty:
            print("No numerical features found for correlation heatmap.")
            return
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(numerical_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title("Correlation Heatmap")
        plt.show()


    def plot_sentiment_distribution(self):
        sentiment_counts = self.dataframe['sentiment'].value_counts()
        plt.figure(figsize=(8, 5))
        sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="viridis")
        plt.title("Distribution of Sentiments in Headlines")
        plt.xlabel("Sentiment")
        plt.ylabel("Count")
        plt.show()

    # function to plot average price change by sentiment
    def plot_price_changes(self):
        if 'Close' not in self.dataframe.columns or 'Open' not in self.dataframe.columns:
            print("Stock price data (Open, Close) is missing!")
            return

        self.dataframe['price_change'] = self.dataframe['Close'] - self.dataframe['Open']  # Daily price change
        sentiment_groups = self.dataframe.groupby('sentiment')['price_change'].mean().reset_index()

        plt.figure(figsize=(8, 5))
        sns.barplot(
            data=sentiment_groups,
            x='sentiment',
            y='price_change',
            palette="coolwarm",
        )
        plt.title("Average Stock Price Change by Sentiment")
        plt.xlabel("Sentiment")
        plt.ylabel("Average Price Change")
        plt.axhline(0, color='gray', linestyle='--')
        plt.show()

    def plot_category(self):
        # Plot the categories
        plt.figure(figsize=(8, 5))
        sns.barplot(x=self.dataframe.index, y=self.dataframe.values, palette="coolwarm")
        plt.title("Distribution of Daily Returns")
        plt.xlabel("Return Category")
        plt.ylabel("Count")
        plt.show()