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

import prepare_data_for_DB as db_import



img_ffill_limit = 1
img_bfill_limit = 1


def read_data():
    inf_df = db_import.read_join(root_dir)
    summarize_df(inf_df)
    filled_df = fill_df(inf_df)
    filled_df.fillna(value=0, inplace=True)
    summarize_df(filled_df, name='0_fill')
    return filled_df

def summarize_df(df, name='before_fill'):
    fname = os.path.join(root_dir, 'Summaries', f'{home_system}_inf_{name}.txt')
    with open(fname, 'w+') as writer:
        for col in [ 'audio',  'env',  'img',  'occupied']:
            # print(col)
            # print(df[col].value_counts())
            print(df[col].value_counts(normalize=True))
            writer.write(f'\n{col}\n {df[col].value_counts(normalize=True)}')
            writer.write(f'\nnan: {df[col].isnull().sum()/len(df[col])}\n')

            # print(f'{col}\n {df[col].value_counts()/len(df[col]):.2}')
            # print(f'nan: {df[col].isnull().sum()/len(df[col]):.2}\n')
    writer.close()
    print(f'{fname} : Write Sucessful!')


def fill_df(df):
    filled_df = df.copy(deep=True)
    filled_df[['img']] = filled_df[['img']].fillna(method='ffill', limit=img_ffill_limit*360)
    filled_df[['img']] = filled_df[['img']].fillna(method='bfill', limit=img_bfill_limit*360)
    filled_df = filled_df.fillna(0)
    return(filled_df)



def create_pg(df, drop=False):
    # print(df)
    # table_name = f'{H_num.lower()}_{color}_inference'
    table_name = f'{home_parameters["home"]}_inf'

    if drop:
        pg.drop_table(table_name)

    pg.create_table(table_name=table_name)
    pg.insert_table(df=df, table=table_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Description")

    parser.add_argument('-path','--path', default='', type=str, help='path of stored data') # Stop at house level, example G:\H6-black\
    args = parser.parse_args()
    root_dir = args.path

    home_system = os.path.basename(root_dir.strip('/'))
    # H_num, color = home_system.split('-')

    occ_file = os.path.join(root_dir, 'Inference_DB', 'Full_inferences', f'{home_system}_occupancy.csv')
    if not os.path.isfile(occ_file):
        print('No occupancy summary found. Generating CSV...')
        write_occupancy_df(root_dir)
    
    inf_file = os.path.join(root_dir, 'Inference_DB', 'Full_inferences', f'{home_system}_full_inf.csv')
    if not os.path.isfile(inf_file):
        print('Full inference not found. Generating inference CSV....')
        final_df = read_data()
    else:
        print(f'Reading df from: {inf_file}')
        final_df = pd.read_csv(inf_file)
        final_df.drop(columns=['Unnamed: 0'], inplace=True)
        summarize_df(final_df, name='after_read')
        

    for col in ['img', 'audio', 'env', 'occupied']:
        final_df[col] = final_df[col].fillna(0.0).astype(int)

    # home_parameters = {'home': f'{H_num.lower()}_{color}'}
    home_parameters = {'home': home_system.lower().replace('-', '_')}
    pg = PostgreSQL(home_parameters)
    create_pg(final_df, drop=True)

