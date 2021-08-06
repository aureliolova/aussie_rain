DATA_PATH = 'weatherAUS.csv'

DATE_COLS = ['date']

TARGET_COL = 'raintomorrow'

CATEGORICAL_COLS = [
    'location',
    'raintoday' 
]


NUMERIC_COLS = [
    'mintemp',
    'maxtemp', 
    'rainfall',
    'evaporation',
    'sunshine',
    'windgustspeed',
    'windspeed9am',
    'windspeed3pm',
    'humidity9am',
    'humidity3pm',
    'pressure9am',
    'pressure3pm',
    'cloud9am',
    'cloud3pm',
    'temp9am',
    'temp3pm',
]