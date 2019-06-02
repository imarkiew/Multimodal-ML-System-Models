import pandas as pd
import csv


METADATA_PATH = '../data/oryginal_data/HAM10000_metadata.tab'
SEPARATOR = '\t'
STATS_PATH = './stats'


df = pd.read_csv(METADATA_PATH, sep=SEPARATOR, engine='python')
dx = df[['dx']]
nr_of_classes = len(set(dx['dx']))
class_distribution = dx\
    .groupby('dx').size()\
    .reset_index(name='counts')\
    .rename(columns={'dx': 'class'})
class_distribution = pd.merge(class_distribution, df
                              .drop_duplicates(subset='lesion_id')[['dx']]
                              .groupby('dx')
                              .size()
                              .reset_index(name='unique_counts')
                              .rename(columns={'dx': 'class'}), how='inner')
class_description = 'Number of classes: {} \n {} \n counts_sum: {} || unique_counts_sum: {}'\
    .format(nr_of_classes, class_distribution, class_distribution[['counts']].sum(), class_distribution[['unique_counts']].sum())

with open(STATS_PATH, 'w') as csvfile:
    csv_writer = csv.writer(csvfile, delimiter=',')
    csv_writer.writerow(['Number of classes: ' + str(nr_of_classes)])
    csv_writer.writerow(['Counts: ' + str(class_distribution[['counts']].sum()['counts'])])
    csv_writer.writerow(['Unique counts: ' + str(class_distribution[['unique_counts']].sum()['unique_counts'])])

class_distribution.to_csv(STATS_PATH, sep=',', encoding='utf-8', index=False, mode='a')
