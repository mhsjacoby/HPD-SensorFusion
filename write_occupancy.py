

import os
import sys
import csv
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from glob import glob

from my_functions import *


def write_occupancy_df(path):
    home_system = os.path.basename(path.strip('/'))
    H_num, color = home_system.split('-')

    save_path = make_storage_directory(os.path.join(path, 'Inference_DB/Full_inferences/'))
    occupant_files = glob(f'{path}/GroundTruth/*.csv')
    occupant_names = []
    occupants = {}
    
    enter_times, exit_times = [], []

    for occ in occupant_files:
        occupant_name = os.path.basename(occ).split('-')[1].strip('.csv')
        occupant_names.append(occupant_name)
        ishome = []
        with open(occ) as csv_file:
            csv_reader, line_count = csv.reader(csv_file, delimiter=','), 0
            for row in csv_reader:
                status, when = row[1], row[2].split('at')
                dt_day = datetime.strptime(str(when[0] + when[1]), '%B %d, %Y %I:%M%p')
                ishome.append((status, dt_day))
                if line_count == 0:
                    enter_times.append(dt_day)
                line_count += 1
            exit_times.append(dt_day)
        occupants[occupant_name] = ishome
    first, last = sorted(enter_times)[0], sorted(exit_times)[-1]

    occ_range = pd.date_range(start=first, end=last, freq='10S')
    occ_df = pd.DataFrame(index=occ_range)

    for occ in occupants:
        occ_df[occ] = 99
        state1 = 'exited'
        for row in occupants[occ]:
            date = row[1]
            state2 = row[0]
            occ_df.loc[(occ_df.index < date) & (occ_df[occ] == 99) & (state1 == 'exited') & (state2 == 'entered'), occ] = 0
            occ_df.loc[(occ_df.index < date) & (occ_df[occ] == 99) & (state1 == 'entered') & (state2 == 'exited'), occ] = 1
            state1 = state2
        occ_df.loc[(occ_df.index >= date) & (occ_df[occ] == 99) & (state1 == 'exited'), occ] = 0
        occ_df.loc[(occ_df.index >= date) & (occ_df[occ] == 99) & (state1 == 'entered'), occ] = 1
    
    occ_df['occupied'] = occ_df[list(occupants.keys())].max(axis=1)
    occ_df.index = pd.to_datetime(occ_df.index)
    occ_df.index.name = 'timestamp'
    fname = os.path.join(save_path, f'{home_system}_occupancy.csv')
    occ_df.to_csv(fname, index = True)
    print(fname + ': Write Successful!')