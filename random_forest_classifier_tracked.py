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
from sklearn import compose
from sklearn import base

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
def build_encoder() -> prep.FunctionTransformer:
    encoding_dict = {'Yes':1, 'No':0}
    encoder = prep.FunctionTransformer(func=encoding_dict.get)
    return encoder

def treat_target(target:pd.Series) -> ty.Tuple[prep.LabelEncoder, np.ndarray]:
    # encoder = build_encoder()
    # treated_target = encoder.transform(target)
    encoder = {'Yes':1, 'No':0}
    treated_target = target.map(encoder.get).to_numpy()
    return encoder, treated_target

#%%feature treatment for all
def build_selector(columns:str=None):
    def select_columns(data:pd.DataFrame, columns:str=None) -> pd.DataFrame:
        if not columns:
            columns = data.columns
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

def build_generic_feature_pipeline(selector_columns:list, dropper_thresh:float=0) -> pipe.Pipeline:
    selector = build_selector(columns=selector_columns)
    dropper = build_dropper(thresh=dropper_thresh)
    transformer = pipe.Pipeline(steps=[
        ('selector', selector),
        ('dropper', dropper)
    ])
    return transformer

#%% numeric feature treatement
def build_imputer(strategy:str='mean') -> impute.SimpleImputer:
    return impute.SimpleImputer(strategy=strategy)

def build_scaler(scaler:str='StandardScaler', **kwargs) -> base.TransformerMixin:
    transformer = getattr(prep, scaler)(**kwargs)
    return transformer

def build_numeric_feature_pipeline(imputer_strategy:str='mean', scaler:str='StandardScaler') -> pipe.Pipeline:
    imputer = build_imputer(imputer_strategy)
    scaler = build_scaler(scaler)
    transformer = pipe.Pipeline(steps=[
        ('imputer', imputer),
        ('scaler', scaler)
    ])
    return transformer

#%% categorical feature treatement

def build_ohe_encoder() -> prep.OneHotEncoder:
    encoder = prep.OneHotEncoder(sparse=False)
    return encoder

def build_categorical_feature_pipeline() -> pipe.Pipeline:
    ohe_encoder = build_ohe_encoder()
    transformer= pipe.Pipeline(steps=[
        ('ohe', ohe_encoder)
    ])
    return transformer

#%%pipelines

def build_feature_transformer(numeric_columns:list, categorical_columns:list, imputer_strategy:str, scaler:str) -> compose.ColumnTransformer:
    numeric_pipeline = build_numeric_feature_pipeline(imputer_strategy=imputer_strategy, scaler=scaler)
    numeric_selector = compose.make_column_selector(dtype_include=np.number)

    categorical_pipeline = build_categorical_feature_pipeline()
    categorical_selector = compose.make_column_selector(dtype_exclude=np.number)

    feature_transformer = compose.ColumnTransformer(transformers=[
        ('numeric', numeric_pipeline, numeric_selector),
        ('categorical', categorical_pipeline, categorical_selector)],
        remainder='drop'
    )
    return feature_transformer

def build_feature_pipeline(numeric_columns:list=list(), categorical_columns:list=list(), dropper_thresh:float=0, imputer_strategy:str='mean', scaler:str='StandardScaler') -> pipe.Pipeline:
    selector_columns = numeric_columns + categorical_columns
    
    generic_feature_pipeline = build_generic_feature_pipeline(selector_columns=selector_columns, dropper_thresh=dropper_thresh)

    feature_transformer = build_feature_transformer(numeric_columns=numeric_columns, categorical_columns=categorical_columns, imputer_strategy=imputer_strategy, scaler=scaler)
    pipeline = pipe.Pipeline(steps=[
        ('generic', generic_feature_pipeline),
        ('transform', feature_transformer)
    ])
    return pipeline

#%%ML model
def build_regressor(**kwargs) -> ensemble.RandomForestClassifier:
    regressor = ensemble.RandomForestClassifier(**kwargs)
    return regressor

def build_model_pipeline(feature_pipeline:compose.ColumnTransformer, **kwargs) -> pipe.Pipeline:
    regressor = build_regressor(**kwargs)
    regressor_params = regressor.get_params(deep=False)
    model_pipeline = pipe.Pipeline(steps=[
        ('feature_engineering', feature_pipeline),
        ('classifier', regressor)
    ])
    return regressor_params, model_pipeline

def get_experiment(experiment_name:str) -> str:
    experiment = mlflow.get_experiment_by_name(name=experiment_name)
    
    if experiment:
        experiment_id = experiment.experiment_id
    else:
        experiment_id = mlflow.create_experiment(experiment_name)
    return experiment_id

def track_training(classifier:pipe.Pipeline, data_tuple:ty.Tuple[np.ndarray], param_dict:dict):
    experiment_id = get_experiment('random_forests')
    
    f_train, f_test, t_train, t_test = data_tuple

    run_name = 'RFC_OHE_' + dt.datetime.now().strftime('%y%m%d%H%M%S')
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
        mlflow.log_params(params=param_dict)
        # mlflow.log_params(params=classifier.get_params())
        
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

def training_loop(data_tuple:ty.Tuple, feature_pipeline:compose.ColumnTransformer, feature_params:dict):
    logger = lg.getLogger('tracker')
    
    search_space_specs = {
        'n_estimators': np.arange(100, 500, 100, dtype=np.int16),
        'max_depth': [None] + np.arange(100, 350, 50, dtype=np.int16).tolist(),
        'n_jobs':10
    }
    builder = SearchSpaceBuilder(constant_keys=['n_jobs'])
    search_space = builder.build(search_space_specs)

    for ctr, param_dict in enumerate(search_space):
        logger.info('training model {ctr} on parameters: {pdict}'.format(ctr=ctr, pdict=str(param_dict)))
        regressor_params, model = build_model_pipeline(feature_pipeline=feature_pipeline, **param_dict)
        full_params = {**regressor_params, **feature_params}
        track_training(classifier=model, data_tuple=data_tuple, param_dict=full_params)
        logger.info('finished training')



# %%
def main():
    logger = set_up_tracking_logger()
    x, y = get_data(target_col=init.TARGET_COL)

    logger.info('Treating target')
    encoder, target = treat_target(y)
    logger.info('Target treated')
    
    data_tuple = m_sel.train_test_split(x, target, test_size=0.2, random_state=42)

    logger.info('Creating feature pipeline')
    feature_param_dict = {
        'numeric_columns':init.NUMERIC_COLS,
        'categorical_columns':init.CATEGORICAL_COLS,
        'dropper_thresh':0.1,
        'imputer_strategy':'mean',
        'scaler':'StandardScaler'
    }
    feature_pipeline = build_feature_pipeline(**feature_param_dict)
    logger.info('Feature pipeline created')

    logger.info('Starting training')
    training_loop(data_tuple, feature_pipeline=feature_pipeline, feature_params=feature_param_dict)
    logger.info('Training complete') 

# %%
if __name__ == '__main__':
    main()