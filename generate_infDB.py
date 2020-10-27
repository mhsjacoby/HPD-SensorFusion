"""
generate_infDB.py
Author: Maggie Jacoby
Last update: October 26, 2020

Broken off from previous create_DB.py. Now able to create multipe types of databases

"""

import os
import sys
import csv
import argparse
from glob import glob
import numpy as np
import pandas as pd

# from datetime import datetime, timedelta


from my_functions import *
from pg_functions import *
import create_pgDB as genDB
# from pg_functions import *


from write_occupancy import *

# import prepare_data_for_DB as db_import



def prepare_inf_for_DB(path, save_loc=''):
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
        inf_df = prepare_inf_for_DB(root_dir)
        fname_before = os.path.join(root_dir, 'Summaries', f'{home_system}_inf_beforeFill.txt')
        genDB.summarize_DF(inf_df, fname_before)
        filled_df = genDB.fill_inf(inf_df)ß
        # filled_df.fillna(value=0, inplace=True)
        fname_after = os.path.join(root_dir, 'Summaries', f'{home_system}_inf_afterFill.txt')
        genDB.summarize_DF(filled_df, fname_after)


    else:
        print(f'Reading df from: {inf_file}')
        final_df = pd.read_csv(inf_file)
        final_df.drop(columns=['Unnamed: 0'], inplace=True)
        fname_before = os.path.join(root_dir, 'Summarißes', f'{home_system}_inf_beforeRead.txt')
        genDB.summarize_DF(final_df, fname_before)

        final_df.fillna(value=0, inplace=True)
        fname_after = os.path.join(root_dir, 'Summaries', f'{home_system}_inf_afterRead.txt')
        genDB.summarize_DF(final_df, fname_after)

        

    for col in ['img', 'audio', 'env', 'occupied']:
        final_df[col] = final_df[col].fillna(0.0).astype(int)

    # home_parameters = {'home': f'{H_num.lower()}_{color}'}
    home_parameters = {'home': home_system.lower().replace('-', '_')}
    pg = PostgreSQL(home_parameters)
    genDB.create_pg(final_df, pg, home_parameters['home'], drop=False)