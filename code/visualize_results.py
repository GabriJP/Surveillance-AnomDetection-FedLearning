# -*- coding: utf-8 -*-
"""

@author: Nicolás Cubero Torres
@description: Make graphs to visualize the performance measures of a grid of
		anomaly and temporal thresholds values.

@usage: visualize_results.py -r <json file containing the performance measures>
"""

# Imported modules
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt

## Defined function
def get_results_data(data: dict or list):

	"""Gets the list of dicts containing the performance measures for each pair
		anomaly and temporal threshold
	"""

	results = []

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

### Input Arguments
parser = argparse.ArgumentParser(description='Make graphs for the main '\
						' performance metrics stored'\
						' on a json results file for '\
						'each pair of anomaly and '\
						' temporal thresholds')
parser.add_argument('-r', '--results', help='JSON file containing the '\
					' performance metrics', type=str, required=True)

args = parser.parse_args()

results_fn = args.results

# Get base filename
dot_pos = results_fn.rfind('.')
if dot_pos != -1:
	base_fn = results_fn[:dot_pos]
else:
	base_fn = results_fn

## Load the json results file
with open(results_fn) as f:
	try:
		results_data = json.load(f)
	except Exception as e:
		print('Cannot load JSON results file'\
			' :\n',str(e), file=sys.stderr)
		exit(-1)

results = get_results_data(results_data)

## Draw graphs performance
if results:
	results = pd.DataFrame.from_records(results, exclude=('confusion matrix',
													'reconstruction_error_norm'))

	for y in ('accuracy', 'precision',	'recall', 'specificity', 'AUC', 'EER',
																	'f1 score'):
		fig, ax = plt.subplots(1)

		for temp_thresh, group in results.groupby('temp_thresh'):
			group.plot(x='anom_thresh', y=y, ax=ax, label=temp_thresh)

		plt.legend(title='temp threshold')
		plt.savefig(base_fn+'_{}.svg'.format(y))
