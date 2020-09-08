import os
import sys
import csv
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from glob import glob

from my_functions import *


def read_join(path, save_loc=''):
    home_system = os.path.basename(path)
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
