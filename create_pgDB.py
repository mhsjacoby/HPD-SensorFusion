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
import time

from my_functions import *
from write_occupancy import *
from pg_functions import *

# img_ffill_limit = 1
# img_bfill_limit = 1
    

def final_fill(df):
    
    dfs = []

    for hub in df['hub'].unique():
        hub_df = df.loc[df['hub'] == hub]
        hub_df.reset_index(inplace=True)
        hub_df = get_forward_pred(hub_df)
        hub_df.index = hub_df['index']
        hub_df = hub_df.drop(columns = ['index'])
        
        dfs.append(hub_df)
    full_df = pd.concat(dfs)

    return full_df


def get_forward_pred(data):
    df = data.copy()

    time_window = cos_win(min_win=.25, max_win=fill_limit, df_len=len(df))
    # ind_map = {x:y for x, y in zip(df.index, time_window)}   # x is the index number of the df, y is the lookahead value

    cols = ['audio', 'img']
    for col in cols:
        s = time.time()
        changes = {}
        ind_list = df.loc[df[col] == 1].index

        changes['before'] = df[col].value_counts()
        print(f'Setting {len(ind_list)} indices in column {col} on hub {df["hub"].unique()[0]}') 
        

        for idx in ind_list:
            j = idx + time_window[idx]
            df.loc[(df.index >= idx) & (df.index <= j), col] = 1

        changes['after'] = df[col].value_counts()
        
        df[col] = df[col].fillna(-1).astype(int)
        changes['final'] = df[col].value_counts()

        changes_df = pd.DataFrame.from_dict(changes).fillna(0).astype(int)
        e = time.time()
        print(f'Time to complete: {e-s} seconds')
        print(changes_df,'\n')
    
    return df


def cos_win(min_win=.25, max_win=1, df_len=8640):
    min_win = min_win * 360
    max_win = max_win * 360

    win_range = max_win - min_win
    t = np.linspace(0, df_len, df_len)
    win_lim = np.round(win_range/2 * np.cos(t*2*np.pi/8640) + win_range/2 + min_win).astype(int)
    return win_lim


def create_pg(df, db_type, drop=False):
    table_name = f'{home_parameters["home"]}_{db_type}'

    if drop:
        pg.drop_table(table_name)

    if db_type == 'inf':
        pg.create_inf_table(table_name=table_name)

    elif db_type == 'prob':
        pg.create_prob_table(table_name=table_name)

    pg.insert_table(df=df, table=table_name)


def prepare_data_for_DB(path, db_type, save_loc=''):
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
                if db_type == 'prob':
                    f_df = pd.read_csv(day, index_col='timestamp')       
                    cols = [col for col in f_df.columns if 'prob' in col]
                    f_df = f_df[cols]
                    day_dfs.append(f_df)

                if db_type == 'inf':
                    day_dfs.append(pd.read_csv(day, index_col='timestamp', usecols=['timestamp', 'occupied']))

            mod_df = pd.concat(day_dfs)

            if mod == 'env_inf' and db_type == 'prob':
                cols = [col.split('_')[1] for col in mod_df.columns]
                mod_df.columns = cols

            else: 
                mod_df.columns = [mod.split('_')[0]]

            all_mods.append(mod_df)
        all_mods.append(occupancy_df)

        for df in all_mods:
            df.index = pd.to_datetime(df.index)
            df['hub'] = hub
            df.set_index([df.index, 'hub'], inplace=True)

        hub_df = pd.concat(all_mods, axis=1, sort=False)
        hub_df.drop(hub_df.tail(1).index,inplace=True) 
        hub_df.to_csv(os.path.join(save_path, f'{H_num}_{hub}_{db_type}.csv'))    
        all_hubs.append(hub_df) 
        
    df = pd.concat(all_hubs)
    print('********** df before writing csv **********')
    print(df)

    df['date'] = df.index.get_level_values(0)
    df.insert(loc=0, column='day', value=df['date'].dt.date)
    df.insert(loc=1, column='hr_min_sec', value=df['date'].dt.time)
    df.insert(loc=2, column='hub', value=df.index.get_level_values(1))

    df.drop(columns=['date'], inplace=True)
    df.reset_index(inplace=True, drop=True)
    df = df[(df['day'] > days[0]) & (df['day'] < days[-1])]

    df.insert(loc=0, column='entry_id', value = df.index+1)

    df.to_csv(os.path.join(save_path, f'{home_system}_full_{db_type}.csv'))
    
    return(df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Description")

    parser.add_argument('-path','--path', default='', type=str, help='path of stored data') # Stop at house level, example G:\H6-black\
    parser.add_argument('-db_type','--db_type', default='inf', type=str, help='Type of database to create (inference, probability, ...')
    parser.add_argument('-schema', '--schema', default='public', type=str, help='Schema to use (default is public).')
    parser.add_argument('-fill_limit', '--fill_limit', default=2, type=int)
    args = parser.parse_args()
    root_dir = args.path
    db_type = args.db_type
    schema = args.schema
    fill_limit = args.fill_limit

    home_system = os.path.basename(root_dir.strip('/'))
    print(f'Using schema: {schema}. Fill limit: {fill_limit}')

    occ_file = os.path.join(root_dir, 'Inference_DB', 'Full_inferences', f'{home_system}_occupancy.csv')
    if not os.path.isfile(occ_file):
        print('No occupancy summary found. Generating CSV...')
        write_occupancy_df(root_dir)
    
    db_file = os.path.join(root_dir, 'Inference_DB', 'Full_inferences', f'{home_system}_full_{db_type}.csv')
    if not os.path.isfile(db_file):
        print(f'Full inference not found. Generating CSV for: {db_file}  ....')
        final_df = prepare_data_for_DB(root_dir, db_type=db_type)
        
    else:
        print(f'Reading df from: {db_file}')
        final_df = pd.read_csv(db_file)
        final_df.drop(columns=['Unnamed: 0'], inplace=True)
    
    final_df = final_fill(final_df)

    home_parameters = {'home': home_system.lower().replace('-', '_')}
    pg = PostgreSQL(home_parameters, schema=schema)
    create_pg(final_df, db_type, drop=False)

