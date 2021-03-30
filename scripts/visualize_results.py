# -*- coding: utf-8 -*-
"""

@author: Nicolás Cubero Torres
@description: Make graphs to visualize the performance measures of a grid of
        anomaly and temporal thresholds values.

@usage: visualize_results.py -r <json file containing the performance measures>
"""

# Imported modules
import argparse
import json
import sys

import matplotlib.pyplot as plt
import pandas as pd

# Defined mapping
MAP_FIELD_NAMES = {'accuracy': 'Accuracy',
                   'precision': 'Precision',
                   'recall': 'Recall',
                   'specificity': 'Specificity',
                   'AUC': 'AUC',
                   'EER': 'EER',
                   'f1 score': 'F1 Score',
                   'TPRxTNR': 'TPRxTNR'}


# Defined function
def get_results_data(data: dict or list):
    """Gets the list of dicts containing the performance measures for each pair
        anomaly and temporal threshold
    """

    results = list()

    if isinstance(data, list):
        for d in data:
            results.extend(get_results_data(d))
    elif 'results' in data and isinstance(data['results'], list):
        results.extend(data['results'])
    else:
        for key in data:
            if isinstance(data[key], (dict, list)):
                results.extend(get_results_data(data[key]))

    return results


"""
and all(
                                            (key in data['results']) for key in
                                                ('anom_thresh', 'temp_thresh',
                                                'accuracy', 'precission',
                                                    'recall', 'specificity',
                                                        'AUC', 'EER')
"""

# Input Arguments
parser = argparse.ArgumentParser(description='Make graphs for the main '
                                             ' performance metrics stored'
                                             ' on a json results file for '
                                             'each pair of anomaly and '
                                             ' temporal thresholds')
parser.add_argument('-r', '--results', help='JSON file containing the '
                                            ' performance metrics', type=str, required=True)

args = parser.parse_args()

results_fn = args.results

# Get base filename
dot_pos = results_fn.rfind('.')
if dot_pos != -1:
    base_fn = results_fn[:dot_pos]
else:
    base_fn = results_fn

# Load the json results file
with open(results_fn) as f:
    try:
        results_data = json.load(f)
    except Exception as e:
        print('Cannot load JSON results file :\n', str(e), file=sys.stderr)
        exit(-1)

results = get_results_data(results_data)

# Draw graphs performance
if results:
    results = pd.DataFrame.from_records(results, exclude=('confusion matrix', 'reconstruction_error_norm'))

    for y in MAP_FIELD_NAMES.keys():

        if y not in results:
            continue

        # Plot the metric
        fig, ax = plt.subplots(1)

        for temp_thresh, group in results.groupby('temp_thresh'):
            group.plot(x='anom_thresh', y=y, ax=ax, label=temp_thresh)

        plt.legend(title=u'\u03BB')
        plt.ylabel(MAP_FIELD_NAMES[y])
        plt.xlabel(u'\u03BC')
        plt.title(MAP_FIELD_NAMES[y])
        plt.savefig(f'{base_fn}_{y}.pdf')
