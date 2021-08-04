import init

import sys
from utils.utilities import *

import mlflow
import sklearn as sk
from sklearn import preprocessing as prep
from sklearn import model_selection as selection     


def na_drop(pdf:pd.DataFrame, na_thresh:float) -> pd.DataFrame:
    '''
    Drop columns from pdf where the percent of NaN values are above na_thresh.
    :param pdf: the DataFrame
    :param na_thresh: the threshold
    :return: tuple(pdf after dropping, column names that where dropped)
    '''

    int_thresh = math.ceil(na_thresh * pdf.shape[0])
    na_counts = pdf.isna().sum()
    drop_cols = na_counts[na_counts >= int_thresh].index
    clean_pdf = pdf.drop(columns = drop_cols)
    return clean_pdf, drop_cols

def group_impute(data:pd.DataFrame, groups:list, subset:list=None, strategy:str='mean'):
    '''
    Impute missing data by groups
    - data: pd.DataFrame to impute
    - subset: columns to impute, default None; if None, all numeric columns will be imputed.
    - strategy: {'mean', 'median', 'mode'} imputation strategy.
    - groups: list of grouping column names.

    Returns:
    - imputed_data: data after imputation.
    '''
    if isinstance(groups, str):
        groups = [groups]
    if isinstance(subset, str):
        subset = [subset]

    dtypes = [np.dtype('float16'), np.dtype('float32'), np.dtype('float64'), np.dtype('int16'), np.dtype('int32'), np.dtype('int64')]
    if subset is None:
        types = data.dtypes
        subset = types[types.isin(dtypes)].index
    else:
        types = data.loc[:, subset].dtypes
        if not all(types.isin(dtypes)):
            raise TypeError(f'some of the columns in {subset} are not numeric.')

    imp_data = data.loc[:, groups + subset]
    rest_data = data.drop(columns=subset)
    
    grouped_imp = imp_data.groupby(groups, as_index=False)

    if strategy == 'mean':
        trans_func = lambda col: col.fillna(col.mean())
    elif strategy == 'median':
        trans_func = lambda col: col.fillna(col.median())
    elif strategy == 'mode':
        trans_func = lambda col: col.fillna(col.mode().mean())
    else: 
        raise ValueError(f'invalid strategy {strategy}')
    
    imputed_values = grouped_imp.transform(trans_func)
    
    imputed_data = pd.merge(left=rest_data, right=imputed_values, left_index=True, right_index=True, validate='1:1')
    return imputed_data


def read_data():
    logger = lg.getLogger('tracker')

    raw_data = pd.read_csv(init.DATA_PATH)
    data = lower_columns(raw_data)
    data_col_set = set(data.columns)
    init_col_set = set(init.NUMERICAL_COLS + init.DATE_COLS + init.CATEGORICAL_COLS) | {init.TARGET_COL}
    
    if not data_col_set >= init_col_set:
        raise ValueError('the data column set contains columns not in initial column set: {}'.format(data_col_set - init_col_set))
    elif not init_col_set >= data_col_set:
        raise ValueError('the initial column set contains columns not in data column set: {}'.format(init_col_set - data_col_set))
    
    data.loc[:, init.DATE_COLS] = pd.to_datetime(data.loc[:, init.DATE_COLS])
    logger.info('read raw data successfully; shape: {}'.format(data.shape))
    return data

def clean_data(data:pd.DataFrame, na_thresh:float,) -> pd.DataFrame:
    logger = lg.getLogger('tracker')
    nan_drop_df, drop_cols = na_drop(pdf=data, na_thresh=na_thresh)
    logger.info(f'dropped columns: {", ".join(drop_cols)}')
    return nan_drop_df, drop_cols

def impute_data(pdf:pd.DataFrame):
    pass

def main():
    '''
    Execute prep pipeline:
    Reading
    '''
    set_up_tracking_logger()
    data = read_data()
    return data


if __name__ == '__main__':
    main()









