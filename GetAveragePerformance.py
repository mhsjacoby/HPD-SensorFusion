"""
GetAveragePerformance.py
Author: Maggie Jacoby
Date: 2021-01-28
"""

import os
import sys
import csv
import json
import argparse
import itertools
import numpy as np
import pandas as pd
from glob import glob
from datetime import datetime, timedelta, time
# from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

from my_functions import *

start_end_file = 'start_end_dates.json'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Get average performance across all hubs.")

    parser.add_argument('-path','--path', default='/Users/maggie/Desktop/InferenceDB', type=str, help='path of stored data')
    args = parser.parse_args()

    root_dir = args.path
    
    start_end_file = 'start_end_dates.json'
    


    for home_path in sorted(glob(os.path.join(root_dir, 'H*'))):

        home_system = os.path.basename(home_path.strip('/'))
        H_num, color = home_system.split('-')
        print(home_system)
        all_days = get_date_list(read_file=start_end_file, H_num=H_num)

        occ_file = os.path.join(home_path, 'Full_inferences', f'{home_system}_occupancy.csv')
        occ_df = pd.read_csv(occ_file, index_col='timestamp', usecols=['timestamp', 'occupied'])

        hub_paths = sorted(glob(f'{home_path}/Full_inferences/{H_num}_*S*.csv'))

        all_hubs = []
    
        audio_list, img_list, env_list = [], [], []
        for h_path in hub_paths:
            print(h_path)
            # hub_df = pd.read_csv(h_path, index_col='timestamp', usecols=['timestamp', 'audio', 'img'] )
            hub_df = pd.read_csv(h_path, index_col='timestamp', usecols=['timestamp', 'audio', 'img', 'env'] )
    
            audio_list.append(hub_df['audio'])
            img_list.append(hub_df['img'])
            env_list.append(hub_df['env'])


        audio_df = pd.concat(audio_list, axis=1)
        max_audio = audio_df.max(axis=1)
        max_audio.name = 'Audio'

        img_df = pd.concat(img_list, axis=1)
        max_img = img_df.max(axis=1)
        max_img.name = 'Image'

        env_df = pd.concat(env_list, axis=1)
        max_env = env_df.max(axis=1)
        max_env.name = 'Env'

        # df = pd.concat([max_audio, max_img, occ_df], axis=1)
        df = pd.concat([max_audio, max_img, max_env, occ_df], axis=1)
        df.insert(loc=2, column='AudImg', value=df[['Audio', 'Image']].max(axis=1))

        df['date'] = df.index
        df['date'] = pd.to_datetime(df['date']) 
        df.insert(loc=0, column='day', value=df['date'].dt.date)
        all_days_datetime = [datetime.strptime(d, '%Y-%m-%d').date() for d in all_days]
        df_daysIn = df.day.isin(all_days_datetime)
        df = df[df_daysIn]

        df.drop(columns=['date', 'day'], inplace=True)
        df.index.name = 'timestamp'
        
        print(df.isna().sum())
        # print(df)
        fname = os.path.join(root_dir, 'SimpleSummaries', f'{H_num}_infSummary.csv')
        df.to_csv(fname)

