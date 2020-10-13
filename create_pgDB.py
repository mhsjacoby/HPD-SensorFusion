"""
create_pgDB.py
Author: Maggie Jacoby
Last update: October 13, 2020

Similar to the Jupyter notebook Load_data_into_Postgres.ipynb, which takes in the 
inferences (audio_inf, img_ing, env_inf) and creates a postgreSQL database from them.
checks to see if ...occupancy.csv exists and if ...inf.csv exists, if so, reads them, if not, creates them

"""

# import psycopg2
# import psycopg2.extras as extras

import os
import sys
import csv
from glob import glob
import argparse

import numpy as np
import pandas as pd

from datetime import datetime, timedelta

# from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
# from functools import reduce

from my_functions import *
from pg_functions import *
from write_occupancy import *

import prepare_data_for_DB as db_import
# import run_ffa as FFA



img_ffill_limit = 6
img_bfill_limit = 1


def read_data():
    inf_df = db_import.read_join(root_dir)
    summarize_df(inf_df)

    """ Use filled_df if filling to simulate time persistency, else use inf_df_dumb_fill """
    filled_df = fill_df(inf_df)
    summarize_df(filled_df, name='after_fill')
    # filled_df = inf_df.fillna(0)
    # summarize(filled_df, name='dumb_fill')
    return filled_df

def summarize_df(df, name='before_fill'):
    fname = os.path.join(root_dir, 'Summaries', f'{home_system}_inf_{name}_summary.txt')
    with open(fname, 'w+') as writer:
        for col in [ 'audio',  'env',  'img',  'occupied']:
            writer.write(f'\n{col}\n {df[col].value_counts()}')
            writer.write(f'nan: {df[col].isnull().sum()/len(df):.2}\n')

            print(f'{col}\n {df[col].value_counts()}')
            print(f'nan: {df[col].isnull().sum()/len(df):.2}\n')
    writer.close()
    print(f'{fname} : Write Sucessful!')


def fill_df(df):
    filled_df = df.copy(deep=True)
    filled_df[['img']] = filled_df[['img']].fillna(method='ffill', limit=img_ffill_limit*360)
    filled_df[['img']] = filled_df[['img']].fillna(method='bfill', limit=img_bfill_limit*360)
    filled_df = filled_df.fillna(0)
    return(filled_df)



def create_pg(df, drop=False):
    print(df)
    table_name = f'{H_num.lower()}_{color}_inference_filled'
    if drop:
        pg.drop_table(table_name)

    pg.create_table(t_name='_inference_filled')
    pg.insert_table(df=df, table=table_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Description")

    parser.add_argument('-path','--path', default='', type=str, help='path of stored data') # Stop at house level, example G:\H6-black\
    args = parser.parse_args()
    root_dir = args.path

    home_system = os.path.basename(root_dir.strip('/'))
    H_num, color = home_system.split('-')

    occ_file = os.path.join(root_dir, 'Inference_DB', 'Full_inferences', f'{home_system}_occupancy.csv')
    if not os.path.isfile(occ_file):
        print('No occupancy sumamry found. Generating CSV...')
        write_occupancy_df(root_dir)
    
    inf_file = os.path.join(root_dir, 'Inference_DB', 'Full_inferences', f'{home_system}_full_inf.csv')
    if not os.path.isfile(inf_file):
        print('Full inference not found. Generating inference CSV....')
        final_df = read_data()
        # final_df.index=final_df['entry_id']
    else:
        print(f'Reading df from: {inf_file}')
        final_df = pd.read_csv(inf_file)#, index_col='entry_id')
        final_df.drop(columns=['Unnamed: 0'], inplace=True)

    for col in ['img', 'audio', 'env', 'occupied']:
        final_df[col] = final_df[col].fillna(0.0).astype(int)

    home_parameters = {'directory': root_dir, 'home': f'{H_num.lower()}_{color}'}
    pg = PostgreSQL(home_parameters)
    create_pg(final_df)

