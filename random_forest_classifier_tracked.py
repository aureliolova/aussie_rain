#%%
import os
import sys

import datetime as dt
import logging as lg
import typing as ty
import math
import copy
import time

import numpy as np
import pandas as pd
import mlflow
from mlflow import sklearn as mlsk
from sklearn import preprocessing as prep
from sklearn import pipeline as pipe
from sklearn import model_selection as m_sel
from sklearn import metrics
from sklearn import impute
from sklearn import ensemble

from build_search_space import SearchSpaceBuilder

#%%
import init
import utilities as u

#%%
def set_up_tracking_logger(log_file:str=None, log_level=lg.INFO) -> lg.Logger:
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
    return logger

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
def build_regressor(**kwargs) -> ensemble.RandomForestClassifier:
    regressor = ensemble.RandomForestClassifier(**kwargs)
    return regressor

def get_experiment(experiment_name:str) -> str:
    experiment = mlflow.get_experiment_by_name(name=experiment_name)
    
    if experiment:
        experiment_id = experiment.experiment_id
    else:
        experiment_id = mlflow.create_experiment(experiment_name)
    return experiment_id

def track_training(classifier:ensemble.RandomForestClassifier, data_tuple:ty.Tuple[np.ndarray], param_dict:dict):
    experiment_id = get_experiment('random_forests')
    
    f_train, f_test, t_train, t_test = data_tuple

    run_name = 'RBC_' + dt.datetime.now().strftime('%y%m%d%H%M%S')
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
        mlflow.log_params(params=param_dict)
        mlflow.log_params(params=classifier.get_params())
        
        model = classifier.fit(X=f_train, y=t_train)
        t_pred = model.predict(f_test)

        metric_dict = {
            'precision':metrics.precision_score(y_true=t_test, y_pred=t_pred),
            'recall':metrics.recall_score(y_true=t_test, y_pred=t_pred)
        }
        
        mlflow.log_metrics(metrics=metric_dict)
        artifact_path = mlflow.get_artifact_uri().lstrip('file:///').replace('%20', ' ')
        mlsk.log_model(sk_model=model, artifact_path=artifact_path)
    return model

def training_loop(data_tuple:ty.Tuple, feature_params:dict):
    logger = lg.getLogger('tracker')
    
    search_space_specs = {
        'n_estimators':np.arange(100, 600, 100, dtype=np.int16),
        'max_depth':[None] + np.arange(100, 350, 50, dtype=np.int16).tolist(),
        'n_jobs':10
    }
    builder = SearchSpaceBuilder(constant_keys=['n_jobs'])
    search_space = builder.build(search_space_specs)

    for ctr, param_dict in enumerate(search_space):
        logger.info('training model {ctr} on parameters: {pdict}'.format(ctr=ctr, pdict=str(param_dict)))
        regressor = build_regressor(**param_dict)
        track_training(classifier=regressor, data_tuple=data_tuple, param_dict=feature_params)
        logger.info('finished training')



# %%
def main():
    logger = set_up_tracking_logger()
    x, y = get_data(target_col=init.TARGET_COL)

    logger.info('Treating target')
    encoder, target = treat_target(y)
    logger.info('Target treated')

    logger.info('Fitting feature pipeline')
    feature_param_dict = {
        'selector_columns':init.NUMERIC_COLS,
        'dropper_thresh':0.1,
        'imputer_strategy':'mean'
    }
    feature_pipeline = build_feature_pipeline(**feature_param_dict)
    features = feature_pipeline.fit_transform(x)
    logger.info('Feature pipeline fit')
    
    logger.info('Starting training')
    data_tuple = m_sel.train_test_split(features, target, test_size=0.2, random_state=42)
    training_loop(data_tuple, feature_params=feature_param_dict)
    logger.info('Training complete') 

# %%
if __name__ == '__main__':
    main()