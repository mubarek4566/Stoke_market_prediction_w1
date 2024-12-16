import pandas as pd 
import os 

# Get the current working directory
current_dir = os.getcwd()

# Build the relative path

def get_path_news():
    # Path to the folder containing CSV files
    file_path = os.path.join(current_dir, '../data/raw_analyst_ratings.csv')  
    return file_path

def get_path_price():
    # Path to the folder containing CSV files
    folder_path = os.path.join(current_dir, '../data/yfinance_data')
    # folder_path = 'C:/Users/Admin/Week1_data/yfinance_data'  
    return folder_path

def new_load(path):
    return path

