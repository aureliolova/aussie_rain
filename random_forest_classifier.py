#%%
import os
import sys

import datetime as dt
import logging as lg
import typing as ty
import math
import copy

import numpy as np
import pandas as pd
from sklearn import preprocessing as prep
from sklearn import pipeline as pipe
from sklearn import model_selection as m_sel
from sklearn import impute
from sklearn import ensemble


#%%
import init
import utilities as u

#%%
def set_up_tracking_logger(log_file:str=None, log_level=lg.INFO):
    logger = lg.getLogger('tracker')
    logger.setLevel(log_level)

    formatter = lg.Formatter('%(module)s @ %(funcName)s :: %(message)s')

    if log_file:
        handler = lg.FileHandler(log_file)
    else:
        handler = lg.StreamHandler(sys.stdout)

    handler.setLevel(log_level)
    handler.setFormatter(formatter)

    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(handler)

#%%
def read_data():
    logger = lg.getLogger('tracker')

    raw_data = pd.read_csv(init.DATA_PATH)
    data = u.lower_columns(raw_data)
    data_col_set = set(data.columns)
    init_col_set = set(init.NUMERIC_COLS + init.DATE_COLS + init.CATEGORICAL_COLS) | {init.TARGET_COL}
    
    if not data_col_set >= init_col_set:
        raise ValueError('the data column set contains columns not in initial column set: {}'.format(data_col_set - init_col_set))
    elif not init_col_set >= data_col_set:
        raise ValueError('the initial column set contains columns not in data column set: {}'.format(init_col_set - data_col_set))
    
    for col in init.DATE_COLS:
        data.loc[:, col] = pd.to_datetime(data.loc[:, col])

    logger.info('read raw data successfully; shape: {}'.format(data.shape))
    return data

def drop_unknown_targets(data:pd.DataFrame, target_col:str) -> pd.DataFrame:
    logger = lg.getLogger('tracker')
    clean_data = (
        data
        .dropna(axis=0, how='any', subset=[target_col])
        .reset_index(drop=True)
    )
    
    logger.info('dropped {} rows with null target value.'.format(len(data) - len(clean_data)))
    return clean_data

def split_features_and_target(data:pd.DataFrame, target_col:str) -> ty.Tuple[pd.DataFrame, pd.Series]:    
    target = data.loc[:, target_col]
    features = data.drop(columns=[target_col])
    return features, target

def get_data(target_col:str) -> ty.Tuple[pd.DataFrame, pd.Series]:
    data = read_data()
    clean_data = drop_unknown_targets(data=data, target_col=target_col)
    X, y = split_features_and_target(data=clean_data, target_col=target_col)
    return X, y

#%%target treatment
def build_encoder() -> prep.LabelEncoder:
    return prep.LabelEncoder()

def treat_target(target:pd.Series) -> ty.Tuple[prep.LabelEncoder, np.ndarray]:
    encoder = build_encoder()
    fitted_encoder = encoder.fit(target)
    treated_target = fitted_encoder.transform(target)
    return fitted_encoder, treated_target

#%%feature treatment
def build_selector(columns:str):
    def select_columns(data:pd.DataFrame, columns:str) -> pd.DataFrame:
        return data.loc[:, columns]
    
    selector = prep.FunctionTransformer(func=select_columns, kw_args={'columns':columns})
    return selector

def build_dropper(thresh:float=0):

    def na_col_drop(data:pd.DataFrame, thresh:float=0) -> pd.DataFrame:
        '''
        Drop columns from data which exceed the threshold.
        If thresh is between 0 and 1 (inclusive) then it is treated as a percentage.
        If thresh is greater than 1 it will be rounded up to the nearest integer and used as a count threshold.
        '''
        logger = lg.getLogger('tracker')

        if thresh < 0:
            raise ValueError('thresh must be positive.')
        elif  0 <= thresh and thresh <= 1:
            thresh = math.ceil((1-thresh)*data.shape[0])
        else:
            thresh = data.shape[0] - math.ceil(thresh)
        
        logger.info(f'thresh: {thresh} {type(thresh)}')
        og_cols = set(data.columns)
        dropped_data = data.dropna(axis=1, thresh=thresh)
        new_cols = set(dropped_data.columns)
        dropped_cols = og_cols - new_cols
        
        logger.info('dropped columns: {}'.format(dropped_cols))
        
        return dropped_data
    
    dropper = prep.FunctionTransformer(func=na_col_drop, kw_args={'thresh':thresh})
    return dropper

def build_imputer(strategy:str='mean') -> impute.SimpleImputer:
    return impute.SimpleImputer(strategy=strategy)


#%%pipelines

def build_feature_pipeline(selector_columns:str, dropper_thresh:float, imputer_strategy:str) -> pipe.Pipeline:
    selector = build_selector(columns=selector_columns)
    dropper = build_dropper(thresh=dropper_thresh)
    imputer = build_imputer(strategy=imputer_strategy)

    pipeline = pipe.Pipeline(steps=[
        ('selector', selector),
        ('dropper', dropper),
        ('imputer', imputer)
    ])
    return pipeline


#%%ML model
def build_regressor() -> ensemble.RandomForestClassifier:
    regressor = ensemble.RandomForestClassifier()
    return regressor

# %%
def main():
    x, y = get_data(target_col=init.TARGET_COL)

    encoder, target = treat_target(y)
    feature_pipeline = build_feature_pipeline(selector_columns=init.NUMERIC_COLS, dropper_thresh=0.1, imputer_strategy='mean')
    features = feature_pipeline.fit_transform(x)

    f_train, f_test, t_train, t_test = m_sel.train_test_split(features, target, test_size=0.2, random_state=42)
    regressor = build_regressor()
    regressor.fit(X=f_train, y=t_train)
    print(regressor.score(f_train, t_train))
    score = regressor.score(X=f_test, y=t_test)
    print(score)
    return regressor

    

# %%
main()