"""
GetConfusion.py
Author: Maggie Jacoby
Date: January 29, 2021
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
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

import matplotlib.pyplot as plt
import seaborn as sns

from my_functions import *


def plot_confusion(conf_mat, title):
    mat = pd.DataFrame(conf_mat, columns = ["Unoccupied", "Occupied"], index = ["Unoccupied", "Occupied"])
    fig, ax = plt.subplots()
    sns.heatmap(mat, annot = True, cbar = False, ax = ax, fmt = "g", square = True, annot_kws = {"fontsize": "x-large"})
    ax.set(xlabel = "Predicted Label", ylabel = "True Label")
    fig.savefig(f'/Users/maggie/Desktop/conf_{title}.png')


def get_metrics(df, case, title):
    
    cm = confusion_matrix(df['occupied'], df[case], labels=[0,1])
    # print(cm)
    # print(cm.ravel())
    tn, fp, fn, tp = cm.ravel()
    acc = accuracy_score(df['occupied'], df[case])
    f1 = f1_score(df['occupied'], df[case])
    total = tp+tn+fp+fn

    tpr = tp/(tp+fn) if tp+fn > 0 else 0.0
    fpr = fp/(tn+fp) if tn+fp > 0 else 0.0

    tnr = tn/(tn+fp) if tn+fp > 0 else 0.0
    fnr = fn/(tp+fn) if tp+fn > 0 else 0.0
    
    pp = tp+fp
    np = tn+fn

    p = tp+fn
    n = tn+fp

    ppr = pp/total
    npr = np/total

    pr = p/total
    nr = n/total


    metrics_df = pd.DataFrame([{'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn, 'Accuracy': acc,
             'TPR': tpr, 'FPR': fpr, 'TNR': tnr, 'FNR': fnr, 'F1': f1,
             'Pos Pred': pp, 'Neg Pred': np, 'Pos Act': p, 'Neg Act': n,
             'Pos Pred Ratio': ppr, 'Neg Pred Ratio': npr, 'Positive Ratio': pr, 'Negative Ratio': nr 
             }])

    plot_
    # mat = pd.DataFrame(cm, columns = ["Unoccupied", "Occupied"], index = ["Unoccupied", "Occupied"])
    # fig, ax = plt.subplots()
    # sns.heatmap(mat, annot = True, cbar = False, ax = ax, fmt = "g", square = True, annot_kws = {"fontsize": "x-large"})
    # ax.set(xlabel = "Predicted Label", ylabel = "True Label")
    # fig.savefig('/Users/maggie/Desktop/conf_mat.png')
    sys.exit()


    return metrics_df





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Get average performance across all hubs.")

    parser.add_argument('-path','--path', default='/Users/maggie/Desktop/InferenceDB/SimpleSummaries', type=str, help='path of stored data')
    args = parser.parse_args()

    root_dir = args.path
    store_loc = make_storage_directory(os.path.join(root_dir, 'Metrics'))

    all_homes = []

    for home_file in sorted(glob(os.path.join(root_dir, 'H*'))):

        H_num = os.path.basename(home_file.strip('/')).split('_')[0]
        # H_num, color = home_system.split('-')
        print(H_num)

        df = pd.read_csv(home_file, index_col='timestamp')
        # print(df)

        # cases = ['No fill', 'fill 0', 'fill 1', 'include Env']


        AudImg_df = df[['occupied', 'AudImg']]
        AudImg_df = AudImg_df.dropna()
        AudImg_nofill = get_metrics(AudImg_df, 'AudImg')
        AudImg_nofill.index = ['Audio & Image (drop nans)']

        AudImg0_df = df[['occupied', 'AudImg']]
        AudImg0_df = AudImg0_df.fillna(0)
        AudImg_fill0 = get_metrics(AudImg0_df, 'AudImg')
        AudImg_fill0.index = ['Audio & Image (0 fill)']


        AudImg1_df = df[['occupied', 'AudImg']]
        AudImg1_df = AudImg1_df.fillna(1)
        AudImg_fill1 = get_metrics(AudImg1_df, 'AudImg')
        AudImg_fill1.index = ['Audio & Image (1 fill)']

        if H_num == 'H4':
            metrics_df = pd.concat([AudImg_nofill, AudImg_fill0, AudImg_fill1], axis=0)

        else:
            Env_df = df[['occupied', 'Env']]
            Env_df = Env_df.dropna()
            Env_nofill = get_metrics(Env_df, 'Env')
            Env_nofill.index = ['Environmental (drop nans)']

            
            metrics_df = pd.concat([AudImg_nofill, AudImg_fill0, AudImg_fill1, Env_nofill], axis=0)
        metrics_df['Home'] = H_num
        metrics_df.index.name = 'Cases Investigated'
        metrics_df = metrics_df.set_index(['Home', metrics_df.index])
        # metrics_df.drop(columns=['Home'])
        print(metrics_df)

        all_homes.append(metrics_df)

    all_homes_df = pd.concat(all_homes, axis=0)
    all_homes_df = all_homes_df.sort_index(level='Cases Investigated')
    all_homes_df.to_csv(os.path.join(store_loc, 'all_homes_summary.csv'))




    