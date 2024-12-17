# import Python libraries
import pandas as pd
import os
from path import get_csv_path

# Define data loader class
class CSVData:
    def __init__(self, folder_path):
        # Initialize the Folder path of the data
        self.folder_path = folder_path
        self.datas = []

    def load_data_files(self):
        csv_files = [f for f in os.listdir(self.folder_path) if f.endswith('.csv')]        
        if not csv_files:
            print("No CSV files found in the folder!")
        
        # Loop through each CSV file and load it into a DataFrame
        for csv_file in csv_files:
            file_path = os.path.join(self.folder_path, csv_file) 
            df = pd.read_csv(file_path) 
            # Append the dataframe to the list
            self.datas.append(df)
    
    def merge_dataframes(self):
        # Merge all csv files into one
        merged_df = pd.concat(self.datas, ignore_index=True)
        return merged_df
    
    def load_csv_file(self):
        """
        Function to load a CSV file using the path returned by get_csv_path().
        """
        csv_path = get_csv_path()
        try:
            data = pd.read_csv(csv_path)
            return data
        except FileNotFoundError:
            print(f"Error: File not found at {csv_path}. Please check the path.")
            return None
        except Exception as e:
            print(f"An error occurred while reading the file: {e}")
            return csv_path

    def load_news_csv(file_path):
        """Loads a single CSV file from a given path and converts it to a DataFrame."""
        try:
            # Check if the file exists and is a CSV
            if not os.path.isfile(file_path) or not file_path.endswith('.csv'):
                raise ValueError("The provided file path is invalid or not a CSV file.")
            
            # Load the CSV file into a DataFrame
            df = pd.read_csv(file_path)
            
            # Check if the DataFrame is empty
            if df.empty:
                print(f"Warning: The file at {file_path} is empty.")
                return None
            return df
        
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            return None
