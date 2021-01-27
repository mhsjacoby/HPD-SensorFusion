import os
import sys
import csv
import argparse
import numpy as np
from glob import glob

from my_functions import *

from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.formatting.rule import DataBarRule
from openpyxl.styles.colors import Color


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Description")

    parser.add_argument('-path','--path', default='', type=str, help='path of stored data') # Stop at house level, example G:\H6-black\
    args = parser.parse_args()
    file_path = args.path
    

    home_system = os.path.basename(file_path.strip('/')).split('_')[0]
    view_set = os.path.basename(file_path.strip('/')).split('_')[1].strip('.csv')
    run_comparison = os.path.basename(file_path.strip('/')).split('_')[-1].strip('.csv')
    root_dir = os.path.split(file_path.rstrip('/'))[0]

    metrics = pd.read_csv(file_path)

    