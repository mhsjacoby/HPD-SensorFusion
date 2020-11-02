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

# import prepare_data_for_DB as db_import



img_ffill_limit = 1
img_bfill_limit = 1


def read_data():
    inf_df = prepare_data_for_DB(root_dir)
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




def prepare_data_for_DB(path, save_loc=''):
    home_system = os.path.basename(path.strip('/'))
    H_num, color = home_system.split('-')

    save_path = make_storage_directory(save_loc) if len(save_loc) > 0 else make_storage_directory(os.path.join(path, 'Inference_DB/Full_inferences'))
    hub_paths = sorted(glob(f'{path}/Inference_DB/{color[0].upper()}S*'))
    all_hubs = []
    
    for h_path in hub_paths:
        hub = os.path.basename(h_path)

        occupancy = []        
        occ_file_paths = glob(f'{path}/Inference_DB/Full_inferences/*_occupancy.csv')
        if len(occ_file_paths) > 1:
            print(f'Too many occupancy files {len(occ_file_paths)}. Exiting.')
            sys.exit()
        occ = occ_file_paths[0]
        occupancy_df = pd.read_csv(occ, index_col='timestamp',  usecols=['timestamp', 'occupied'])

        idx = pd.to_datetime(occupancy_df.index)
        days = sorted(set([x.date() for x in idx]))
        
        all_mods = []
        mods = mylistdir(h_path, bit='_inf', end=True)
        for mod in mods:
            dates = sorted(glob(f'{h_path}/{mod}/*.csv'))
            print(hub, mod, len(dates))

            day_dfs = []
            
            for day in dates:
                # print(os.path.basename(day).strip('.csv'))
                day_dfs.append(pd.read_csv(day, index_col='timestamp', usecols=['timestamp', 'occupied']))
                
            mod_df = pd.concat(day_dfs)
            mod_df.columns = [mod.split('_')[0]]
            all_mods.append(mod_df)
        all_mods.append(occupancy_df)
        
        for df in all_mods:
            df.index = pd.to_datetime(df.index)
            df['hub'] = hub
            df.set_index([df.index, 'hub'], inplace=True)

        hub_df = pd.concat(all_mods, axis=1, sort=False)
        hub_df.drop(hub_df.tail(1).index,inplace=True) 
        hub_df.to_csv(os.path.join(save_path, f'{H_num}_{hub}_inf.csv'))    
        all_hubs.append(hub_df) 
        
    df = pd.concat(all_hubs)
    
    df['date'] = df.index.get_level_values(0)
    df.insert(loc=0, column='day', value=df['date'].dt.date)
    df.insert(loc=1, column='hr_min_sec', value=df['date'].dt.time)
    df.insert(loc=2, column='hub', value=df.index.get_level_values(1))

    df.drop(columns=['date'], inplace=True)
    df.reset_index(inplace=True, drop=True)
    df = df[(df['day'] > days[0]) & (df['day'] < days[-1])]

    df.insert(loc=0, column='entry_id', value = df.index+1)

    df.to_csv(os.path.join(save_path, f'{home_system}_full_inf.csv'))
    
    return(df)




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

