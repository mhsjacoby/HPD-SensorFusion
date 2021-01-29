"""
create_localDB.py
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




def prepare_data_for_DB(path, occ_df):
    
    H_num, color = os.path.basename(path.strip('/')).split('-')
    save_path = make_storage_directory(os.path.join(path, 'Full_inferences'))
    hub_paths = sorted(glob(f'{path}/{color[0].upper()}S*'))

    all_hubs = []

    for h_path in hub_paths:
        hub = os.path.basename(h_path)

        idx = pd.to_datetime(occ_df.index)
        days = sorted(set([x.date() for x in idx]))
        
        all_mods = []
        mods = mylistdir(h_path, bit='_inf', end=True)
        for mod in mods:
            dates = sorted(glob(f'{h_path}/{mod}/*.csv'))

            day_dfs = []
            for day in dates:
                day_dfs.append(pd.read_csv(day, index_col='timestamp', usecols=['timestamp', 'occupied']))
                
            mod_df = pd.concat(day_dfs, axis=0)
            mod_df.columns = [mod.split('_')[0]]
            all_mods.append(mod_df)

        all_mods.append(occ_df)
        df = pd.concat(all_mods, axis=1)
        df.index.name = 'timestamp'

        df['date'] = df.index
        df['date'] = pd.to_datetime(df['date']) 
        df.insert(loc=0, column='day', value=df['date'].dt.date)
        df = df.dropna(subset=['occupied'])
        all_df_days = sorted(df.day.unique())
        day1, dayn = all_df_days[0], all_df_days[-1]
        print(len(all_df_days), day1, dayn)

        df = df.drop(df[(df.day == day1) | (df.day == dayn)].index)
        df.drop(columns=['day', 'date'], inplace=True)

        print(df.isna().sum())

        fname = os.path.join(save_path, f'{H_num}_{hub}.csv')
        df.to_csv(fname)

        print(f'Writing to csv: {fname}')
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Description")

    parser.add_argument('-path','--path', default='/Users/maggie/Desktop/InferenceDB', type=str, help='path of stored data')
    args = parser.parse_args()
    root_dir = args.path

    for home_path in sorted(glob(os.path.join(root_dir, 'H*'))):

        home_system = os.path.basename(home_path.strip('/'))
        print(home_system)

        occ_file = os.path.join(home_path, 'Full_inferences', f'{home_system}_occupancy.csv')
        if not os.path.isfile(occ_file):
            print('No occupancy summary found. Generating CSV...')
            write_occupancy_df(home_path)
        else:
            print(f'Using occupancy summary: {occ_file}')
        occ_df = read_occupancy_df(home_path)


        db_file = os.path.join(home_path, 'Full_inferences', f'{home_system}_newOcc.csv')
        prepare_data_for_DB(home_path, occ_df)


