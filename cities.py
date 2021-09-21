#%%
import pandas as pd
import numpy as np

from functools import partial
#%% get data
CITIES_NAME = 'au.csv'
SELECTED_COLUMNS = ['city', 'lat', 'lng']
DATASET_NAME = 'weatherAUS.csv'

def get_data(filename:str) -> pd.DataFrame:
    raw_data = pd.read_csv(filename)
    return raw_data

def format_data(data:pd.DataFrame, columns:list) -> pd.DataFrame:
    data.columns = [col.lower() for col in data.columns]
    data = data.loc[:, columns]
    data.rename(columns={'city':'location'}, inplace=True)
    return data

def subset_to_dataset(loc_data:pd.DataFrame) -> pd.DataFrame:
    feature_dataset = pd.read_csv(DATASET_NAME)
    feature_dataset.columns = [col.lower() for col in feature_dataset.columns]
    location_data = feature_dataset.loc[:, ['location']].drop_duplicates()
    subset_data = pd.merge(left=loc_data, right=location_data, on=['location'], how='inner')
    return subset_data


def get_city_data():
    raw_data = get_data(CITIES_NAME)
    data = format_data(data=raw_data, columns=SELECTED_COLUMNS)
    subset_data = subset_to_dataset(loc_data=data)
    return subset_data

def cross_join(loc_data:pd.DataFrame) -> pd.DataFrame:
    cmp_data = loc_data.copy()
    cmp_data.columns = ['{}_cmp'.format(col) for col in cmp_data.columns]
    loc_data.loc[:, 'key'] = 1
    cmp_data.loc[:, 'key'] = 1
    cross_data = pd.merge(left=loc_data, right=cmp_data, on=['key']).drop(columns=['key'])
    return cross_data

def compute_distance(data:pd.DataFrame, x_y_cols:list) -> pd.DataFrame:
    x_y_comp_cols = ['{}_cmp'.format(col) for col in x_y_cols]

    xy = data.loc[:, x_y_cols].to_numpy()
    xy_comp = data.loc[:, x_y_comp_cols].to_numpy()

    xy_diff = xy - xy_comp
    xy_diff_sq = xy_diff ** 2

    dist_sq = xy_diff_sq.sum(axis=1)
    dist = np.sqrt(dist_sq)
    
    data.loc[:, 'dist'] = dist
    return data

def _select_top(grouped_data:pd.DataFrame, top:int) -> pd.DataFrame:
    sorted_data = grouped_data.sort_values(by=['location', 'dist'], ascending=True)
    selected_data = sorted_data.iloc[1:top+1, :]
    selected_data.loc[:, 'top_label'] = ['top_{}'.format(x) for x in range(1, top+1)]
    return selected_data

def select_top(data:pd.DataFrame, top:int=3) -> pd.DataFrame:
    select_function = partial(_select_top, top=top)
    selected_data = data.groupby('location', as_index=False).apply(select_function).reset_index(drop=True)
    return selected_data

#%%
def write_data(data:pd.DataFrame, filename:str='loc_relation.csv') -> None:
    data.to_csv(filename, index=False)

#%%
def main():
    data = get_city_data()
    cross_data = cross_join(loc_data=data)
    dist_data = compute_distance(data=cross_data, x_y_cols=['lat', 'lng'])
    top_data = select_top(data=dist_data, top=3)
    write_data(top_data)
    return top_data

#%%
if __name__ == '__main__':
    main()

