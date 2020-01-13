import pandas as pd
from data_preprocessing import *

def combine(text_path):
    li = []
    for path in text_path:
        df2 = read_files(path)
        li.append(df2)
    df = pd.concat(li, axis=0, ignore_index=True)
    return df
