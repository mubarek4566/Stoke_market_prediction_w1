# import Python libraries
import pandas as pd
import os

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
    
    def merge_csvdata(self):
        # Merge all csv files into one
        merged_df = pd.concat(self.datas, ignore_index=True)
        return merged_df
    
    def load_news_data(self, file_path):
        # Load the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        return df    

