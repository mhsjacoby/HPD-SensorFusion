"""
create_pgDB.py
Author: Maggie Jacoby
Last update: January 28, 2021
    - Read new occupancy files and input into pg

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



def read_occupancy_df(path):
       
    occ_file_paths = glob(f'{path}/Full_inferences/*_occupancy.csv')
    if len(occ_file_paths) > 1:
        print(f'Too many occupancy files {len(occ_file_paths)}. Exiting.')
        sys.exit()
    occ = occ_file_paths[0]
    print('Reading ', occ)
    occupancy_df = pd.read_csv(occ, index_col='timestamp',  usecols=['timestamp', 'occupied'])
    return occupancy_df




def prepare_data_for_DB(path, occ_df, db_type='inf'):
    
    # home_system = os.path.basename(path.strip('/'))
    H_num, color = os.path.basename(path.strip('/')).split('-')
    save_path = make_storage_directory(os.path.join(path, 'Full_inferences'))
    hub_paths = sorted(glob(f'{path}/{color[0].upper()}S*'))
    # sys.exit()
    all_hubs = []

    for h_path in hub_paths:
        hub = os.path.basename(h_path)

        idx = pd.to_datetime(occ_df.index)
        days = sorted(set([x.date() for x in idx]))
        
        all_mods = []
        mods = mylistdir(h_path, bit='_inf', end=True)
        for mod in mods:
            dates = sorted(glob(f'{h_path}/{mod}/*.csv'))
            # print(hub, mod, len(dates))
            day_dfs = []
            for day in dates:
                # print('Reading', day)
                day_dfs.append(pd.read_csv(day, index_col='timestamp', usecols=['timestamp', 'occupied']))
                
            mod_df = pd.concat(day_dfs, axis=0)
            
            mod_df.columns = [mod.split('_')[0]]
            # print(mod_df)

            all_mods.append(mod_df)
        all_mods.append(occupancy_df)
        print(all_mods)
        sys.exit()


    #         day_dfs = []
            
    #         # for day in dates:
    #         #     if db_type == 'prob':
    #         #         f_df = pd.read_csv(day, index_col='timestamp')       
    #         #         cols = [col for col in f_df.columns if 'prob' in col]
    #         #         f_df = f_df[cols]
    #         #         day_dfs.append(f_df)

    #             if db_type == 'inf':
    #                 print('Reading', day)
    #                 day_dfs.append(pd.read_csv(day, index_col='timestamp', usecols=['timestamp', 'occupied']))
    #                 break

    #         mod_df = pd.concat(day_dfs)

    #         if mod == 'env_inf' and db_type == 'prob':
    #             cols = [col.split('_')[1] for col in mod_df.columns]
    #             mod_df.columns = cols

    #         else: 
    #             mod_df.columns = [mod.split('_')[0]]

    #         all_mods.append(mod_df)
    #     all_mods.append(occupancy_df)

    #     for df in all_mods:
    #         df.index = pd.to_datetime(df.index)
    #         df['hub'] = hub
    #         df.set_index([df.index, 'hub'], inplace=True)

    #     hub_df = pd.concat(all_mods, axis=1, sort=False)
    #     hub_df.drop(hub_df.tail(1).index,inplace=True) 
    #     hub_save_fname = os.path.join(save_path, f'{H_num}_{hub}_{db_type}.csv')
    #     hub_df.to_csv(hub_save_fname)
    #     print(f'Writing to csv: {hub_save_fname}')    
    #     all_hubs.append(hub_df) 
        
    # df = pd.concat(all_hubs)
    # print('********** df before writing csv **********')
    # print(df)

    # df['date'] = df.index.get_level_values(0)
    # df.insert(loc=0, column='day', value=df['date'].dt.date)
    # df.insert(loc=1, column='hr_min_sec', value=df['date'].dt.time)
    # df.insert(loc=2, column='hub', value=df.index.get_level_values(1))

    # df.drop(columns=['date'], inplace=True)
    # df.reset_index(inplace=True, drop=True)
    # df = df[(df['day'] > days[0]) & (df['day'] < days[-1])]

    # df.insert(loc=0, column='entry_id', value = df.index+1)

    # df_write_loc = os.path.join(save_path, f'{home_system}_full_{db_type}_newtest.csv')
    # df.to_csv(df_write_loc)
    # print(f'Writing df to: {df_write_loc}')
    # sys.exit()
    # return(df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Description")

    parser.add_argument('-path','--path', default='/Users/maggie/Desktop/InferenceDB', type=str, help='path of stored data') # Stop at house level, example G:\H6-black\
    parser.add_argument('-db_type','--db_type', default='inf', type=str, help='Type of database to create (inference, probability, ...')
    parser.add_argument('-schema', '--schema', default='new_occ', type=str, help='Schema to use (default is public).')
    parser.add_argument('-fill_limit', '--fill_limit', default=5, type=int)
    args = parser.parse_args()

    root_dir = args.path
    # db_type = args.db_type
    # schema = args.schema
    # fill_limit = args.fill_limit

    for home_path in sorted(glob(os.path.join(root_dir, 'H*'))):
        # print(home_path)
        home_system = os.path.basename(home_path.strip('/'))
        print(home_system)

        occ_file = os.path.join(home_path, 'Full_inferences', f'{home_system}_occupancy.csv')
        if not os.path.isfile(occ_file):
            print('No occupancy summary found. Generating CSV...')
            write_occupancy_df(home_path)
        else:
            print(f'Using occuapncy summary: {occ_file}')
        occ_df = read_occupancy_df(home_path)
        # sys.exit()


        db_file = os.path.join(home_path, 'Full_inferences', f'{home_system}_newOcc.csv')
        if not os.path.isfile(db_file):
            print(f'Full inference not found. Generating CSV for: {db_file}  ....')
            final_df = prepare_data_for_DB(home_path, occ_df)
            
        else:
            print(f'Reading df from: {db_file}')
            final_df = pd.read_csv(db_file)
            final_df.drop(columns=['Unnamed: 0'], inplace=True)
        
        
        print(f'NANS!!!!!! (before)\n: {final_df.isna().sum()}')
        # final_df = final_fill(final_df, fill_limit=5)
        # print(f'NANS!!!!!! (after)\n: {final_df.isna().sum()}')


        # home_parameters = {'home': home_system.lower().replace('-', '_')}
        # pg = PostgreSQL(home_parameters, schema=schema)
        # create_pg(final_df, db_type, drop=True)