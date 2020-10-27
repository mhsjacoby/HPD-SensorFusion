"""
create_pgDB.py
Author: Maggie Jacoby
Last update: October 13, 2020

Similar to the Jupyter notebook Load_data_into_Postgres.ipynb, which takes in the 
inferences (audio_inf, img_ing, env_inf) and creates a postgreSQL database from them.
checks to see if ...occupancy.csv exists and if ...inf.csv exists, if so, reads them, if not, creates them

"""

import os
import sys
import csv
from glob import glob
import argparse

import numpy as np
import pandas as pd

from datetime import datetime, timedelta

from my_functions import *
from pg_functions import *
from write_occupancy import *



img_ffill_limit = 1
img_bfill_limit = 1


def fill_inf(df):
    filled_df = df.copy(deep=True)
    filled_df[['img']] = filled_df[['img']].fillna(method='ffill', limit=img_ffill_limit*360)
    filled_df[['img']] = filled_df[['img']].fillna(method='bfill', limit=img_bfill_limit*360)
    filled_df.fillna(value=0, inplace=True)
    return(filled_df)



def summarize_DF(df, fname, cols=[ 'audio',  'env',  'img',  'occupied']):
    print(fname)
    with open(fname, 'w+') as writer:
        for col in cols:
            print(df[col].value_counts(normalize=True))
            writer.write(f'{col}\n {df[col].value_counts(normalize=True)}')
            writer.write(f'nan: {df[col].isnull().sum()/len(df[col])}\n')
            print(f'nan: {df[col].isnull().sum()/len(df[col]):.2}\n')
    writer.close()
    print(f'{fname} : Write Sucessful!')







def create_pg(df, pg, home, drop=False):
    table_name = f'{home}_inf'

    if drop:
        pg.drop_table(table_name)

    pg.create_table(table_name=table_name)
    pg.insert_table(df=df, table=table_name)


