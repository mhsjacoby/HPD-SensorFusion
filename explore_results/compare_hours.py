"""
compare_hours.py
Author: Maggie Jacoby
Date: November 2020

This is used in exploratory analysis and determines the error metrics by hour in each home.
The class is based on the classes from run_ff.py, but combines instance and home (only one run done).
Predicts with an 'OR' gate using ALL hubs audio and images.
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

from functools import reduce
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

from my_functions import *
from pg_functions import *

from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.formatting.rule import DataBarRule
from openpyxl.styles.colors import Color

class Home():
    
    def __init__(self, pg, system, threshold='0.8'):
        self.pg = pg
        self.system = system.lower().split('-')
        self.pg_system = pg.home
        self.hubs = self.get_distinct_from_DB('hub')
        self.days = self.get_days(threshold, system)

        self.df = self.create_df()
        self.predictions = self.get_predictions()
        self.results_by_hr = self.test_days(days=self.days)

    def get_days(self, threshold, system):
        all_days_db = [x.strftime('%Y-%m-%d') for x in self.get_distinct_from_DB('day')]
        days_above_file = os.path.join('/Users/maggie/Desktop/CompleteSummaries', f'all_days_above_{threshold}.json')
        with open(days_above_file) as file:
            fdata = json.load(file)
        all_days_th = fdata[system]
        days = sorted(list(set(all_days_db).intersection(all_days_th)))
        print(days)
        return days


    def get_distinct_from_DB(self, col):
        print(self.pg_system)

        query = """
            SELECT DISTINCT %s
            FROM %s_inf;
            """ %(col, self.pg_system)

        distinct = self.pg.query_db(query)[col].unique()
        return sorted(distinct)
    
    
    def select_from_hub(self, hub, mod):

        select_query = """
            SELECT day, hr_min_sec, hub, %s, occupied
            FROM %s_inf
            WHERE %s_inf.hub = '%s'
            """ %(mod, self.pg_system, self.pg_system, hub)

        return select_query



    def create_df(self):
        df_list = []
        for hub in self.hubs:
            for mod in ['audio', 'img']:
                print(f'hub: {hub}, modality: {mod}')
                hub_df = self.pg.query_db(self.select_from_hub(hub, mod))
                hub_df.drop(columns=['hub'], inplace=True)
                rename_cols = {mod:f'{mod}_{hub[2]}'}
                hub_df.rename(columns = rename_cols, inplace=True)
                df_list.append(hub_df)

        df_merged = reduce(lambda  left, right: pd.merge(left, right, on=['day', 'hr_min_sec', 'occupied'], how='outer'), df_list)
        
        col = df_merged.pop('occupied')
        df_merged.insert(len(df_merged.columns), col.name, col)
        return df_merged

    
    def get_predictions(self):
        df = self.df.copy()
        skip_cols = ['day', 'hr_min_sec', 'occupied']
        df['prediction'] = 0
        df.loc[df[df.columns.difference(skip_cols)].sum(axis=1) > 0, 'prediction'] = 1
        skip_cols.append('prediction')
        return df[skip_cols]
        

    def test_days(self, days):
        hours = [x for x in range(0,24)]
        results_by_hr = {}
        for day in days:
            print(day)
        for hr in hours:
            TPR, FPR, TNR, FNR, acc = [], [], [], [], []

            tr_start = f'{hr:02}:00:00'
            tr_end = f'{hr:02}:59:50'
            dt_start = datetime.strptime(tr_start, '%H:%M:%S').time()
            dt_end = datetime.strptime(tr_end, '%H:%M:%S').time()

            for day_str in sorted(days):
                day = datetime.strptime(day_str, '%Y-%m-%d').date()
                day_df = self.predictions.loc[self.predictions['day'] == day]
                hr_df = day_df.loc[(day_df['hr_min_sec'] >= dt_start) & (day_df['hr_min_sec'] <= dt_end)]

                tn, fp, fn, tp = confusion_matrix(hr_df['occupied'], hr_df['prediction'], labels=[0,1]).ravel()
                acc.append(accuracy_score(hr_df['occupied'], hr_df['prediction']))

                # tpr = tp/(tp+fn) if tp+fn > 0 else 0.0
                # fpr = fp/(tn+fp) if tn+fp > 0 else 0.0

                # tnr = tn/(tn+fp) if tn+fp > 0 else 0.0
                # fnr = fn/(tp+fn) if tp+fn > 0 else 0.0

                # TPR.append(float(f'{tpr:.4}'))
                # FPR.append(float(f'{fpr:.4}'))

                # TNR.append(float(f'{tnr:.4}'))
                # FNR.append(float(f'{fnr:.4}'))

                TPR.append(tp)
                FPR.append(fp)

                TNR.append(tn)
                FNR.append(fn)


            TPR, FPR, TNR, FNR, acc = np.mean(TPR), np.mean(FPR), np.mean(TNR), np.mean(FNR), np.mean(acc) 
            results_by_hr[hr] = {'TPR': TPR, 'FPR': FPR, 'TNR': TNR, 'FNR': FNR, 'accuracy': acc}

        return results_by_hr



def create_workbook(df):
    wb = Workbook()
    ws = wb.active

    for r in dataframe_to_rows(df, index=True, header=True):
        if r != [None]:
            ws.append(r)
    ws['A1'] = 'Hour'


    for col in ["B2:B25", "D2:D25", "F2:F25"]:
        ws.conditional_formatting.add(col, make_databar(col, "00008000"))
    for col in ["C2:C25", "E2:E25"]:
        ws.conditional_formatting.add(col, make_databar(col, "00FF0000"))
    for col in ["G2:G25", "I2:I25"]:
        ws.conditional_formatting.add(col, make_databar(col, "0099CCFF"))
    for col in ["H2:H25", "J2:J25"]:
        ws.conditional_formatting.add(col, make_databar(col, "00993366"))

    wb.save(f'/Users/maggie/Desktop/openpyxl/{home_system}.xlsx')


def make_databar(col, bar_color):
    data_bar_rule = DataBarRule(start_type="num",
                                start_value=1,
                                end_type="num",
                                end_value=360,
                                color=bar_color
                                )
    return data_bar_rule


def make_sum_cols(df):
    df['Actual Occ'] = df['TPR'] + df['FNR']
    df['Actual Not'] = df['TNR'] + df['FPR']
    df['Pred Occ'] = df['TPR'] + df['FPR']
    df['Pred Not'] = df['TNR'] + df['FNR']
    return df




if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Description")

    parser.add_argument('-system', '--system', type=str)
    args = parser.parse_args()

    home_system = args.system
    H_num, color = home_system.split('-')

    home_parameters = {'home': f'{H_num.lower()}_{color}'}
    pg = PostgreSQL(home_parameters)

    H = Home(pg=pg, system=home_system)
    hr_df = pd.DataFrame(H.results_by_hr)
    
    hr_df = hr_df.transpose()
    hr_df = make_sum_cols(hr_df)
    # print(hr_df)
    create_workbook(hr_df)
    # hr_df.to_csv(f'/Users/maggie/Desktop/CompleteSummaries/{home_system}_hrCompare_RAW.csv')
