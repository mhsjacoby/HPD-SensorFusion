"""
run_ffa.py
Author: Maggie Jacoby
Edited: 2020-10-20 - calculate variance of runs

These classes are used to generate objects for each home, and individual FFA runs. 
Import into Load_data_into_pstgres notebook or run stand alone (preferred).
"""

import os
import sys
import csv
import argparse
import itertools
import numpy as np
import pandas as pd
from glob import glob
from datetime import datetime, timedelta

from functools import reduce
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

from my_functions import *

from pg_functions import *




class Home():
    
    def __init__(self, pg, system, level):
        self.pg = pg
        self.system = system.lower().split('-')
        self.pg_system = pg.home
        self.level = level
        self.hubs = self.get_distinct_from_DB('hub')
        self.days = [x.strftime('%Y-%m-%d') for x in self.get_distinct_from_DB('day')]
        self.run_specifications = self.get_FFA_output()

        
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


    def get_FFA_output(self):
        spec_path = f'/Users/maggie/Documents/Github/HPD-Inference_and_Processing/SensorFusion/fracfact_output'
        num_hubs = len(self.hubs)
        run_file = os.path.join(spec_path, f'{num_hubs}_hub_{self.level}.csv' )

        run_specifications = []
        with open(run_file) as FFA_output:
            for i, row in enumerate(csv.reader(FFA_output), 1):
                run_specifications.append((i, row))
        print(run_specifications)
        return run_specifications



class FFA_instance():
    mod_dict = {'-1': 'audio', '1': 'img'}
    
    def __init__(self, run, Home):
        self.Home = Home
        self.run  = run[0]
        self.spec = run[1]
        self.check_spec()
        self.run_modalities = self.get_hub_modalities()
        self.df = self.create_df()
        self.predictions = self.get_predictions()
        self.results_by_day = {}
        self.rate_results = self.test_days(days=self.Home.days)
        
        self.TPR, self.FPR = np.mean(self.rate_results['TPR']), np.mean(self.rate_results['FPR'])
        self.TNR, self.FNR = np.mean(self.rate_results['TNR']), np.mean(self.rate_results['FNR']) 
        self.f1, self.accuracy = np.mean(self.rate_results['f1']), np.mean(self.rate_results['accuracy'])

        self.var_TPR, self.var_FPR = np.var(self.rate_results['TPR']), np.var(self.rate_results['FPR'])
        self.var_TNR, self.var_FNR = np.var(self.rate_results['TNR']), np.var(self.rate_results['FNR']) 
        self.var_f1, self.var_accuracy = np.var(self.rate_results['f1']), np.var(self.rate_results['accuracy'])
        

    def check_spec(self):
        if len(self.spec) != len(self.Home.hubs):
            print(f'Incorrect run specification. {len(self.Home.hubs)} hubs and {len(self.spec)} spots in FFA.')
        else:
            print(f'All good. Ready to run number {self.run}: {self.spec}.')   
        
    def get_hub_modalities(self):
        run_mods = {}
        for x,y in zip(self.Home.hubs, self.spec):
            run_mods[x] = self.mod_dict[y]
        print(run_mods)
        return run_mods

    
    def create_df(self):
        df_list = []
        for hub in self.run_modalities:
            mod = self.run_modalities[hub]
            print(f'hub: {hub}, modality: {mod}')
            hub_df = self.Home.pg.query_db(self.Home.select_from_hub(hub, mod))
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
        TPR, FPR, TNR, FNR, f1, acc = [], [], [], [], [], []
        
        for day_str in sorted(days):
            day = datetime.strptime(day_str, '%Y-%m-%d').date()
            day_df = self.predictions.loc[self.predictions['day'] == day]

            tn, fp, fn, tp = confusion_matrix(day_df['occupied'], day_df['prediction'], labels=[0,1]).ravel()
            f1.append(f1_score(day_df['occupied'], day_df['prediction']))
            acc.append(accuracy_score(day_df['occupied'], day_df['prediction']))
            self.results_by_day[day_str] = (tn, fp, fn, tp)
            
            tpr = tp/(tp+fn) if tp+fn > 0 else 0.0
            fpr = fp/(tn+fp) if tn+fp > 0 else 0.0

            tnr = tn/(tn+fp) if tn+fp > 0 else 0.0
            fnr = fn/(tp+fn) if tp+fn > 0 else 0.0

            TPR.append(float(f'{tpr:.4}'))
            FPR.append(float(f'{fpr:.4}'))

            TNR.append(float(f'{tnr:.4}'))
            FNR.append(float(f'{fnr:.4}'))

        return {'TPR': TPR, 'FPR': FPR, 'TNR': TNR, 'FNR': FNR, 'f1': f1, 'accuracy': acc}


def get_instances(H):

    d = {
        'Run': [], 'Inclusion': [], 'Name': [],
        'False Positive Rate': [], 'True Positive Rate': [],
        'False Negative Rate': [], 'True Negative Rate': [],
        'F1-Score': [], 'Accuracy': []
        }

    V = {
        'False Positive Rate': [], 'True Positive Rate': [],
        'False Negative Rate': [], 'True Negative Rate': [],
        'F1-Score': [], 'Accuracy': []
        }


    all_instances = {}

    for x in H.run_specifications:
        inst = FFA_instance(x, H)
        
        all_instances[inst.run] = inst
        
        d['False Positive Rate'].append(inst.FPR)
        d['True Positive Rate'].append(inst.TPR)
        d['False Negative Rate'].append(inst.FNR)
        d['True Negative Rate'].append(inst.TNR)
        
        d['Run'].append(inst.run)
        d['Inclusion'].append(inst.spec)
        d['F1-Score'].append(inst.f1)
        d['Accuracy'].append(inst.accuracy)
        d['Name'].append(f'Run {inst.run}: {inst.run_modalities}')


        V['False Positive Rate'].append(inst.var_FPR)
        V['True Positive Rate'].append(inst.var_TPR)
        V['False Negative Rate'].append(inst.var_FNR)
        V['True Negative Rate'].append(inst.var_TNR)
        V['F1-Score'].append(inst.var_f1)
        V['Accuracy'].append(inst.var_accuracy) 

    N_runs = len(H.run_specifications)
    N_days = len(H.days)

    SE = {}
    for v in V:
        print('var', v, np.min(V[v]), np.max(V[v]), np.mean(V[v]))
        print('rate', v, np.min(d[v]), np.max(d[v]), np.mean(d[v]))
        SE[v] = np.sqrt(np.mean(V[v])*(N_days/N_runs))/10
    print(SE) 


    SE_df = pd.DataFrame(SE, index=['SE'])
    roc_df = pd.DataFrame(d)

    return roc_df, SE_df



if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Description")

    # parser.add_argument('-path','--path', default='', type=str, help='path of stored data') # Stop at house level, example G:\H6-black\
    parser.add_argument('-system', '--system', type=str)
    parser.add_argument('-level', '--level', type=str, default='full')
    args = parser.parse_args()

    home_system = args.system
    H_num, color = home_system.split('-')
    run_level = args.level

    home_parameters = {'home': f'{H_num.lower()}_{color}'}
    pg = PostgreSQL(home_parameters)
    H = Home(pg=pg, system=home_system, level=run_level)

    roc_df, SE = get_instances(H)
    df2 = pd.DataFrame(roc_df['Inclusion'].to_list(), columns=H.hubs)

    df2.index = roc_df.index
    df2 = df2.merge(roc_df, left_index=True, right_index=True)
    df2.index = df2['Run']
    df2.drop(columns=['Inclusion', 'Run'], inplace=True)
    dfwSE = df2.append(SE, ignore_index=False)
    dfwSE.index.rename('Run')


    # all_runs_df = pd.DataFrame.from_dict(roc_df)
    # all_runs_df.to_csv(f'/Users/maggie/Desktop/FFA_output/roc_df_{home_system}_{run_level}.csv')

    # var_df.to_csv(f'/Users/maggie/Desktop/FFA_output/varianvce_{home_system}_{run_level}.csv')
    dfwSE.to_csv(f'/Users/maggie/Desktop/FFA_output/{home_system}_{run_level}.csv', index_label='Run')