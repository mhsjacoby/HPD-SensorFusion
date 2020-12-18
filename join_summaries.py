"""
join_summaries.py
Author: Maggie Jacoby
Date: November 2020

This is used to summarize the completeness of each day/hub, and combine summaries for audio and images.
No inputs are needed when running the script.

Outputs summaries (csv) for all days/hubs, plus json with list of days above threshold (default 80%)

"""

import os
import sys
import csv
import argparse
import numpy as np
import pandas as pd
from glob import glob
from datetime import datetime, timedelta, time

from my_functions import *
import json
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

from openpyxl.styles import PatternFill, colors
from openpyxl.styles.differential import DifferentialStyle
from openpyxl.formatting.rule import CellIsRule
from openpyxl.utils.cell import get_column_letter

""" Run Parameters """
perc_threshold = 0.8
systems = ['H1-red', 'H2-red', 'H3-red', 'H4-red', 'H5-red', 'H6-black']
write_loc = make_storage_directory('/Users/maggie/Desktop/CompleteSummaries/2020-12-18')



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

def read_env_summary(filepath):
    df = pd.read_csv(filepath, sep=' ', header=None)
    df.columns = ['hub', 'date', 'in', 'out', 'perc']
    df.index = df['date']
    df.drop(columns = df.columns.difference(['perc']), inplace=True)
    return df


def format_xlsx(workbook, df, home_system):
    ws = workbook.create_sheet(home_system, -1)
    grn = PatternFill(bgColor='00CCFFCC')

    for row in dataframe_to_rows(df, index=True, header=True):
        ws.append(row)
    ws['A1'] = ws['A2'].value
    ws.delete_rows(idx=2, amount=1)

    perc_rule = CellIsRule(operator='greaterThan', formula=[perc_threshold], stopIfTrue=False, fill=grn)
    ws.conditional_formatting.add('B2:Q100', perc_rule)
    return workbook



if __name__ == '__main__':

    days_above_threshold = {}
    workbook = Workbook()
    # sheet = workbook.active

    for home_system in systems:

        print(home_system)
        H_num, color = home_system.split('-')
        
        path = os.path.join(f'/Users/maggie/Desktop/HPD_mobile_data/HPD-env/HPD_mobile-{H_num}/{home_system}')
        hubs = sorted(mylistdir(path, bit=f'{color.upper()[0]}S', end=False))

        cols = [f'{hub}_{m}' for hub in hubs for m in ['A', 'I', 'E']]#, 'I-dark']]
        df = pd.DataFrame(columns=cols)

        dfs = []

        for hub in hubs:
            env_filepath = os.path.join(path, 'Summaries', f'{H_num}-{hub}-data-summary.txt')
            env_df = read_env_summary(env_filepath)
            env_df.columns = [f'{hub}_E']
            dfs.append(env_df)
            
            audio_filepath = os.path.join(path, 'Summaries', f'{H_num}-{hub}-audio-summary.txt')
            audio_df = read_audio_summary(audio_filepath)
            audio_df.columns = [f'{hub}_A']
            dfs.append(audio_df)

            img_filepath = os.path.join(path, 'Summaries', f'{H_num}-{hub}-img-summary.txt')
            img_df = read_img_summary(img_filepath)
            img_df.columns = [f'{hub}_I']#, f'{hub}_I-dark']
            dfs.append(img_df)

        df = pd.concat(dfs, axis=1)
        df = df.sort_index()
        df = df.fillna(0)

        df['Minimum'] = df.min(axis=1)
        days_above = list(df.loc[df['Minimum'] >= perc_threshold].index)
        days_above_threshold[home_system] = days_above
        
        workbook = format_xlsx(workbook=workbook, df=df, home_system=home_system)

        # df_mods = df.drop(df.filter(regex='dark').columns, axis=1, inplace=False)
        # df_mods['Minimum'] = df_mods.min(axis=1)

        # days_above = list(df_mods.loc[df_mods['Minimum'] >= perc_threshold].index)
        # days_above_threshold[home_system] = days_above

    

    workbook.save(os.path.join(write_loc, f'all_sheets_test.xlsx'))
    # full_write_loc = os.path.join(write_loc, f'all_days_above_{perc_threshold}.json')

    # with open(full_write_loc, 'w') as file:
    #     file.write(json.dumps(days_above_threshold)) 
    #     print(f'File written to: {full_write_loc} for {len(days_above_threshold)} homes.')
