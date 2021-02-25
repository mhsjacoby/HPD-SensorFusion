"""
join_probs.py
Author: Maggie Jacoby
Date: 2021-02-24

Read in the modality level inferences (with probablities) and join them together
to create one file for each hub with: img, audio, all envs, ground truth

Similar to part of the code in create_pgDB.py, but without creating the database
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


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Create files with joined probability inferences')

    parser.add_argument('-path','--path', default='', type=str, help='path of stored data') # Stop at house level, example G:\H6-black\
    parser.add_argument('-db_type','--db_type', default='inf', type=str, help='Type of database to create (inference, probability, ...')
    parser.add_argument('-fill_limit', '--fill_limit', default=2, type=int)

    args = parser.parse_args()
    root_dir = args.path
    db_type = args.db_type
    fill_limit = args.fill_limit

    home_system = os.path.basename(root_dir.strip('/'))
    H_num, color = home_system.split('-')
    path = make_storage_directory(os.path.join(root_dir, 'Inference_DB'))
    save_path = make_storage_directory(os.path.join(root_dir, 'Inference_DB', 'Full_inferences'))
    hub_paths = sorted(glob(f'{path}/{color[0].upper()}S*'))
    all_hubs = []

    for h_path in hub_paths:
        hub = os.path.basename(h_path)
        occupancy = []        
        occ_file_paths = glob(f'{path}/Full_inferences/*_occupancy.csv')
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
                f_df = pd.read_csv(day, index_col='timestamp')       
                cols = [col for col in f_df.columns if 'prob' in col]
                f_df = f_df[cols]
                day_dfs.append(f_df)


            mod_df = pd.concat(day_dfs)

            if mod == 'env_inf':
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
