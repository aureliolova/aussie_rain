#%%
import pandas as pd

#%% get data
CITIES_NAME = 'au.csv'
SELECTED_COLUMNS = ['city', 'lat', 'lng']

def get_data(filename:str) -> pd.DataFrame:
    raw_data = pd.read_csv(filename)
    return raw_data

def format_data(data:pd.DataFrame, columns:list) -> pd.DataFrame:
    data.columns = [col.lower() for col in data.columns]
    data = data.loc[:, columns]
    data.rename(columns={'city':'location'}, inplace=True)
    return data

def get_city_data():
    raw_data = get_data(CITIES_NAME)
    data = format_data(data=raw_data, columns=SELECTED_COLUMNS)
    return data
