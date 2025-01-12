# -*- coding: utf-8 -*-
"""

@author: Nicolás Cubero Torres
@description: Several utilities used in the scripts.
"""

# Modules imported
from sys import float_info
import numpy as np
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K

def cum_sum(arr):

	"""Computes the cumulative sum of an indexable collection of values

	   Parameters
	   ----------

	   arr: 1D indexable collection of numeric values
	"""

	# Check input
	if not hasattr(arr, '__getitem__'):
		raise ValueError('"arr" must be indexable')

	if any(not isinstance(x, (float, int)) for x in arr):
		raise ValueError('All collection\'s value must be numeric')

	## Procedure

	cum = [arr[0]]

	for x in arr[1:]:
		cum.append(cum[-1] + x)

	return cum

def make_partitions(arr, *args):

	"""Split an indexable collection of objects into severals partitions
		keeping the passed portions

		Parameters
		---------

		arr : 1D indexable collection of object to be partitionated

		portions : various float values
			Several values indicating the portion of the original data
			collection to be holded on each partition.

			Each portion value must be in [0,1] and the sum of the passed values
			must be 1

		Raise
		-----
		ValueError: if arr doesn't include __getitem__ and __len__ methods
		ValueError: if no portion values
		ValueError: if portion values doesn't sum 1
	"""

	# Check input
	if not (hasattr(arr, '__getitem__') and hasattr(arr, '__len__')):
		raise ValueError('"arr" is not indexable or length cannot be know')

	if not args:
		raise ValueError('Some portion value must be passed')

	if any(not isinstance(x, (float, int)) for x in args):
		raise ValueError('Portion values are not number')

	if np.abs(sum(args) - 1) > float_info.epsilon:
		raise ValueError('portion values doesn\'t sum 1')

	## Procedure

	# Set the portions to cumulative from 0 to 1
	cum_port = [0] + cum_sum(args)[:-1] + [1]

	# Middle splits
	splits = [arr[int(np.round(cum_port[i]*len(arr))):
				int(np.round(cum_port[i+1]*len(arr)))] for i in range(len(cum_port)-1)]

	return splits

def confusion_matrix(y_true, y_pred):

	# Situar vectores de etiqueta en una dimensión
	y_true = y_true.squeeze().astype('int32')
	y_pred = y_pred.squeeze().astype('int32')

	n_classes = y_true.max()+1

	if n_classes < 2:
		n_classes = 2

	# Matriz en la que se almacena el resultado
	conf_matrix = np.zeros((n_classes, n_classes), dtype='int64')

	# Anotar la comparación en la matriz
	for i in range(y_true.size):
		conf_matrix[y_true[i], y_pred[i]] += 1

	return conf_matrix

def equal_error_rate(y_true, y_score, eps=1e-5):

	"""Computes the Equal Error Rate (EER) measure from the binary
		prediction scores.

		Parameters
		----------

		y_true : list, np.ndarray
			The Ground-truth binary labels

		y_score : list, np.ndarray
			The prediction scores set as the probability to the positive class

		eps : float (default: 10^-5)
			Maximum difference between the closest false positive rate and
			false negative rate allowed to consider this point as
			the equal error rate.

			If difference is greater than eps, NaN is returned

		Return
		------
		Tuple of 2 float values:
			- The Equal Error Rate
			- Threshold in which EER is reachen
	"""

	fpr, tpr, threshold = roc_curve(y_true, y_score, pos_label=1)
	fnr = 1 - tpr

	"""
	# Get the difference associated to the closest fnr and fpr points
	eer_fpr = fpr[np.nanargmin(np.abs((fnr - fpr)))]
	eer_fnr = fnr[np.nanargmin(np.abs((fnr - fpr)))]

	return float(eer_fpr if np.abs(eer_fpr - eer_fnr) < eps else np.NaN)
	"""

	# Compute the Equal Error Rate and the threshold in which is reachen
	eer = float(fpr[np.nanargmin(np.abs((fnr - fpr)))])
	eer_thresh = float(threshold[np.nanargmin(np.abs((fnr - fpr)))])

	return eer, eer_thresh

def extract_experiments_parameters(experiment: dict, mult_val_params: list or tuple):

	# Check input
	if not isinstance(experiment, dict):
		raise ValueError('"experiment" must be a dictionnary')

	if (not isinstance(mult_val_params, (list, tuple)) or
					any(not isinstance(x, str) for x in mult_val_params)):
		raise ValueError('"mult_val_params" not a valid list or tuple of str')

	# Separate the parameters which can adopt a list of values from those
	# with only one value
	mult_val_dict = {}
	uni_val_dict = {}

	index_params = {}

	for p in experiment:
		if p in mult_val_params:
			mult_val_dict[p] = experiment[p] if isinstance(experiment[p], (list,
													tuple)) else [experiment[p]]
			index_params[p] = 0
		else:
			uni_val_dict[p] = experiment[p]

	# Store all combination of parameters
	params = []

	if not mult_val_dict:
		return [experiment]

	# Build a list of dictionnaries containing single values for each original
	# multivalue parameters formed as a combination of its values
	keys = list(mult_val_dict.keys())

	while index_params[keys[0]] < len(mult_val_dict[keys[0]]):

		aux = {}

		# Make a combination of multivalue parameters among the univalue parameters
		for k in keys:
			aux[k] = mult_val_dict[k][index_params[k]]

		aux.update(uni_val_dict)
		params.append(aux)

		# Look for another combination of multivalue parameters by incrementing
		# the index of the next multivalue parameters' value to be appended
		index_params[keys[-1]] += 1

		# Complete cycles of previous counters
		for i in range(len(keys)-1,0,-1):

			if index_params[keys[i]] == len(mult_val_dict[keys[i]]):
				index_params[keys[i]] = 0

				# Full cycle of the current multivalue parameter is performed so
				# new combinations with the next previous parameter's value
				# is going to be made
				index_params[keys[i-1]] += 1

	return params

def plot_results(results: dict, metric: str, filename: str):

	plt.figure()

	# Imprimir cada gráfica
	for r in results:
		plt.plot(results[r], label=r)

	# Mostrar leyenda
	plt.title(metric)
	plt.xlabel('epoch')
	plt.ylabel(metric)
	plt.legend(results.keys(), loc='best')#, loc='lower right')

	plt.savefig(filename)

	plt.close() # Cerrar la figura una vez terminado

def root_sum_squared_error(y_true, y_pred):

	if K.ndim(y_true) > 2:
		return K.mean(K.sqrt(K.sum(K.square(y_true - y_pred),
					axis=K.arange(1, K.ndim(y_true)) )))
	else:
		return K.sqrt(K.sum(K.square(y_true - y_pred)))

def split_measures_per_video(meas: np.ndarray,
					cum_cuboids_per_video: list or tuple or np.ndarray):

	"""Separates the measures performed for each test sample
		on different arrays stored on a list with an array per test sample
	"""

	sep = []

	for i in range(len(cum_cuboids_per_video)):

		start = cum_cuboids_per_video[i-1] if i > 0 else 0 # Starting video cuboid
		end = cum_cuboids_per_video[i]						# Ending video cuboid

		sep.append(meas[start: end])

	return sep
