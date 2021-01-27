import sys
import os
import csv
from glob import glob

version = 'V6'

root_dir = '/Users/maggie/Desktop/FFA_output/'
comparisons = ["image-audio", "image-none", "audio-none"]

with open(os.path.join(root_dir, 'version_comparisons', f'{version}_comp.csv'), mode='w') as write_file:
    version_writer = csv.writer(write_file, delimiter=',')
    version_writer.writerow(
        [" ", " ", "False Positive Rate", "True Positive Rate", "False Negative Rate","True Negative Rate", "F1-Score", "Accuracy"])
    for comp in comparisons:
        version_writer.writerow([comp])
        read_files = sorted(glob(os.path.join(root_dir, version, f"*_{comp}.csv")))

        for f in read_files:
            home = os.path.basename(f).split('_')[0]
            with open(f, 'r') as read_file:
                csv_reader = csv.reader(read_file, delimiter = ',')
                line_count = 0
                for row in csv_reader:
                    line_count += 1
                    if line_count == 2:
                        r = [" ", home] + row[1:]
                        version_writer.writerow(r)
                        break


