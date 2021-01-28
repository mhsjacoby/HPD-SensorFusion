"""
calculate_metrics.py
Author: Maggie Jacoby
Date: October 15, 2020


To-Do:
- normalize results
"""


import os
import sys
import csv
import argparse
import itertools
import numpy as np
import pandas as pd
from glob import glob

from my_functions import *
from pg_functions import *

# from openpyxl import Workbook
# from openpyxl.utils.dataframe import dataframe_to_rows
# from openpyxl.formatting.rule import DataBarRule
# from openpyxl.styles.colors import Color



def subset_df(df):
    df.index = df['Run']
    df.drop(columns=['Run'], inplace=True)

    SE = df.loc['SE']
    SE = SE.dropna(axis=0)
    SE_df = pd.DataFrame(data=SE).transpose()

    df.drop(['SE'], inplace=True)

    color = home_system.split('-')[1][0].upper()
    hubs = [col for col in df.columns if f'{color}S' in col]    
    hub_df = df[hubs]

    metrics = [col for col in df.columns if col not in hubs and col != 'Name']
    metric_df = df[metrics]

    name_df = df[['Name']]

    return hub_df, metric_df, name_df, SE_df



def get_effects(hub_df, metric_df):
    full_metrics = []
    div = len(metric_df)/2
    for metric in metric_df.columns:
        new_df = hub_df.multiply(metric_df[metric], axis='index')
        sum_col = new_df.sum(axis=0)/div
        sum_col.rename(metric, inplace=True)
        full_metrics.append(sum_col)
    full_metrics = pd.concat(full_metrics, axis=1)
    return full_metrics



def get_interactions(hub_df, level=2):
    hubs = hub_df.columns
    interactions = list(itertools.combinations(hubs, level))
    interaction_df = []
    for hub_set in interactions:
        col_name = ''
        H1, H2 = hub_set[0], hub_set[1]
        idx = range(1,len(hub_df)+1)
        new_col = pd.Series([1.0]*len(hub_df), index=idx)
        

        for hub in hub_set:
            col_name += hub + 'x'
            mult_col = hub_df[hub]
            new_col.index=mult_col.index
            new_col = new_col.multiply(mult_col)

        col_name = col_name.strip('x')
        new_col.rename(col_name, inplace=True)
        interaction_df.append(new_col)

    interaction_df = pd.concat(interaction_df, axis=1)
    return interaction_df










def avg_effect(df):
    avg_dict = {}
    div = len(df)
    for metric in df.columns:
        # avg = df[metric].sum(axis=1)
        avg_dict[metric] = df[metric].sum()/div
    avg_df = pd.DataFrame(avg_dict, index=['avg'])
    return avg_df




if __name__ == '__main__':

    # parser = argparse.ArgumentParser(description="Description")

    # parser.add_argument('-path','--path', default='', type=str, help='path of stored data') # Stop at house level, example G:\H6-black\
    # args = parser.parse_args()
    # file_path = args.path
    
    file_path = sys.argv[1]
    print(file_path)
    home_system = os.path.basename(file_path.strip('/')).split('_')[0]
    view_set = os.path.basename(file_path.strip('/')).split('_')[1].strip('.csv')
    run_comparison = os.path.basename(file_path.strip('/')).split('_')[-1].strip('.csv')
    root_dir = os.path.split(file_path.rstrip('/'))[0]

    ffa_output = pd.read_csv(file_path)
    hub_df, metric_df, name_df, SE = subset_df(ffa_output)

    Main_effects = get_effects(hub_df, metric_df)
    # Main_effects.to_csv(os.path.join(root_dir, f'{home_system}_main_effects.csv'))

    # TwoFI = get_interactions(hub_df)
    # Two_level_effects = get_effects(TwoFI, metric_df)
    # Two_level_effects.to_csv(os.path.join(root_dir, f'{home_system}_2FI_effects.csv'))

    # ThreeFI = get_interactions(hub_df, level=3)
    # Three_level_effects = get_effects(ThreeFI, metric_df)
    # Three_level_effects.to_csv(os.path.join(root_dir, f'{home_system}_3FI_effects.csv'))

    # full_effects = pd.concat([Main_effects, Two_level_effects, Three_level_effects], axis=0)

    avg = avg_effect(metric_df)
    # full_wavg = pd.concat([avg, full_effects])
    full_wavg = pd.concat([avg, Main_effects])
    Full_metrics = pd.concat([full_wavg, SE])
    # print(Full_metrics)
    save_folder = make_storage_directory('/Users/maggie/Desktop/FFA_output/env_comp')
    Full_metrics.to_csv(os.path.join(save_folder, f'{home_system}_{run_comparison}.csv'))