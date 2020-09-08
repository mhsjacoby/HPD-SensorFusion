"""
run_ffa.py
Author: Maggie Jacoby, August 28 2020

These classes are used to generate objects for each home, and individual FFA runs. Import into Load_data_into_pstgres notebook
"""

import os
import csv
import numpy as np
import pandas as pd
from glob import glob
from datetime import datetime, timedelta

from functools import reduce
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from my_functions import *




class Home():
    
    def __init__(self, path, pg, spec_file=''):
        self.path = path
        self.pg = pg
        system = path.strip('/').split('/')[-1].lower().split('-')
        self.pg_system = f'{system[0]}_{system[1]}'
        self.hubs = self.get_distinct_from_DB('hub')
        self.days = [x.strftime('%Y-%m-%d') for x in self.get_distinct_from_DB('day')]
        self.spec_file = spec_file
        self.run_specifications = self.get_FFA_output(spec_file=self.spec_file)

        
    def get_distinct_from_DB(self, col):

        query = """
            SELECT DISTINCT %s
            FROM %s_inference_filled;
            """ %(col, self.pg_system)

        distinct = self.pg.query_db(query)[col].unique()
        return sorted(distinct)
    
    
    def select_from_hub(self, hub, mod):

        select_query = """
            SELECT day, hr_min_sec, hub, env, %s, occupied
            FROM %s_inference_filled
            WHERE %s_inference_filled.hub = '%s'
            """ %(mod, self.pg_system, self.pg_system, hub)

        return select_query


    def get_FFA_output(self, spec_file, level='L2'):
        spec_path = glob(f'{self.path}/Inference_DB/run_specs/*_{level}.csv') if len(spec_file) == 0 else [os.path.join(f'{self.path}/Inference_DB/run_specs/{spec_file}')]
        run_file = spec_path[0]
        run_specifications = []
        with open(run_file) as FFA_output:
            for i, row in enumerate(csv.reader(FFA_output), 1):
                run_specifications.append((i, row))
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
        self.f1, self.accuracy = np.mean(self.rate_results['f1']), np.mean(self.rate_results['accuracy'])
         
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
            rename_cols = {mod:f'{mod}_{hub[2]}', "env": f'env_{hub[2]}'}
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
        TPR, FPR, f1, acc = [], [], [], []
        
        for day_str in sorted(days):
            day = datetime.strptime(day_str, '%Y-%m-%d').date()
            day_df = self.predictions.loc[self.predictions['day'] == day]

            tn, fp, fn, tp = confusion_matrix(day_df['occupied'], day_df['prediction'], labels=[0,1]).ravel()
            f1.append(f1_score(day_df['occupied'], day_df['prediction']))
            acc.append(accuracy_score(day_df['occupied'], day_df['prediction']) )
            self.results_by_day[day_str] = (tn, fp, fn, tp)
            tpr = tp/(tp+fn) if tp+fn > 0 else 0.0
            fpr = fp/(fp+tn) if fp+tn > 0 else 0.0

            TPR.append(float(f'{tpr:.4}'))
            FPR.append(float(f'{fpr:.4}'))
        return {'TPR': TPR, 'FPR': FPR, 'f1': f1, 'accuracy': acc}