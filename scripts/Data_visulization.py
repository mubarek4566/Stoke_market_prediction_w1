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
        #Univariate Analysis for Numerical Columns Distribution
        numerical_cols = self.dataframe.select_dtypes(include=['int64', 'float64']).columns
        for col in numerical_cols:
            plt.figure()
            sns.histplot(self.dataframe[col], kde=True, bins=30)
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.show()
    
    def univariate_cate(self):
        # Categorical Columns Analysis
        categorical_cols = self.dataframe.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            plt.figure()
            sns.countplot(data=self.dataframe, x=col, order=self.dataframe[col].value_counts().index)
            plt.title(f"Countplot of {col}")
            plt.xlabel(col)
            plt.ylabel("Count")
            plt.xticks(rotation=45)
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

