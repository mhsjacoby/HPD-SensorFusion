"""
join_summaries.py
Author: Maggie Jacoby
Date: November 2020

This is used to summarize the completeness of each day/hub by reading in individual modality summaries
Combines summaries for audio and images for all homes.
No inputs are needed when running the script.

Outputs summaries (csv) for all days/hubs, plus json with list of days above threshold (default 80%)

"""

import os
import sys
import csv
import json
import argparse
import numpy as np
import pandas as pd
from glob import glob
from datetime import datetime, timedelta, time

from my_functions import *

""" Run Parameters """
perc_threshold = 0.8
systems = ['H1-red', 'H2-red', 'H3-red', 'H4-red', 'H5-red', 'H6-black']
write_loc = make_storage_directory('/Users/maggie/Desktop/CompleteSummaries')



def read_audio_summary(filepath):
    df = pd.read_csv(filepath, sep=' ', header=None)
    df.columns = ['hub', 'date', 'in', 'out', 'perc']
    df.index = df['date']
    df.drop(columns = df.columns.difference(['perc']), inplace=True)
    return df

def read_img_summary(filepath):
    df = pd.read_csv(filepath, sep=' ', header=0)
    df.drop(columns = df.columns.difference(['day', '%Capt']), inplace=True)
    df.columns = ['date', 'perc']
    df.index = df['date']
    df.drop(columns = ['date'], inplace=True)
    return df



if __name__ == '__main__':

    days_above_threshold = {}
 
    for home_system in systems:

        print(home_system)
        H_num, color = home_system.split('-')
        
        path = os.path.join(f'/Users/maggie/Desktop/HPD_mobile_data/HPD-env/HPD_mobile-{H_num}/{home_system}')
        hubs = sorted(mylistdir(path, bit=f'{color.upper()[0]}S', end=False))

        cols = [f'{hub}_{m}' for hub in hubs for m in ['A', 'I']]
        df = pd.DataFrame(columns=cols)

        dfs = []

        for hub in hubs:
            audio_filepath = os.path.join(path, 'Summaries', f'{H_num}-{hub}-audio-summary.txt')
            audio_df = read_audio_summary(audio_filepath)
            audio_df.columns = [f'{hub}_A']
            dfs.append(audio_df)

            img_filepath = os.path.join(path, 'Summaries', f'{H_num}-{hub}-img-summary.txt')
            img_df = read_img_summary(img_filepath)
            img_df.columns = [f'{hub}_I']
            dfs.append(img_df)

        df = pd.concat(dfs, axis=1)
        df = df.sort_index()
        df = df.fillna(0)
        df['Minimum'] = df.min(axis=1)
        df.to_csv(os.path.join(write_loc, f'{home_system}_AudioImages.csv'))


        days_above = list(df.loc[df['Minimum'] >= perc_threshold].index)
        days_above_threshold[home_system] = days_above

        full_write_loc = os.path.join(write_loc, f'all_days_above_{perc_threshold}.json')

        with open(full_write_loc, 'w') as file:
            file.write(json.dumps(days_above_threshold)) 
