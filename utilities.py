#%%
import sys
import logging as lg
import pandas as pd

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


def lower_columns(pdf:pd.DataFrame)->pd.DataFrame:
    pdf.columns = [col.lower() for col in pdf.columns]
    return pdf