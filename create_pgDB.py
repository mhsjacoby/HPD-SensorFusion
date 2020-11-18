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
from write_occupancy import *
from pg_functions import *

# img_ffill_limit = 1
# img_bfill_limit = 1
    

def fill_df(df):
    filled_df = df.copy(deep=True)
    # filled_df[['img']] = filled_df[['img']].fillna(method='ffill', limit=img_ffill_limit*360)
    # filled_df[['img']] = filled_df[['img']].fillna(method='bfill', limit=img_bfill_limit*360)
    filled_df[['img']] = filled_df[['img']].fillna(method='ffill', limit=5)
    filled_df[['img']] = filled_df[['img']].fillna(method='bfill', limit=5)
    return(filled_df)



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
    # parser.add_argument('-schema', '--schema', default='public', type=str, help='Schema to use (default is public).')
    args = parser.parse_args()
    root_dir = args.path
    db_type = args.db_type
    # schema = args.schema

    home_system = os.path.basename(root_dir.strip('/'))
    # print(f'Using schema: {schema}')

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
    
    # final_df = fill_df(final_df)
    not_fill_cols = ['entry_id', 'day', 'hr_min_sec', 'hub', 'occupied']
    cols = [col for col in final_df.columns if col not in not_fill_cols]

    for col in cols:
        if db_type == 'inf':
            # final_df[col] = final_df[col].astype(int)

            final_df[col] = final_df[col].fillna(-1).astype(int)
        elif db_type == 'prob':
            final_df[col] = final_df[col].fillna(-1.0)
    print('********** final df to enter into database **********')
    print(final_df)

    home_parameters = {'home': home_system.lower().replace('-', '_')}
    pg = PostgreSQL(home_parameters)#, schema)
    create_pg(final_df, db_type, drop=True)

