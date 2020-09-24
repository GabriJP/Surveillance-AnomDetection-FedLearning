# -*- coding: utf-8 -*-
"""

@author: Nicolás Cubero Torres
@description: Evaluate the prediction of an ISTL model for a given test dataset.

@usage: evaluate_ISTL_detailed.py -m <Pretrained model's h5 file>
					-t <Path folder containing the test video>
"""
# Modules imported
import os
import sys
import json
import argparse
import numpy as np
from cv2 import imread, imwrite, resize, cvtColor, COLOR_BGR2GRAY
import tensorflow
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from models import istl
from utils import root_sum_squared_error, split_measures_per_video

if tensorflow.__version__.startswith('1'):
	from tensorflow import ConfigProto, Session

	config = ConfigProto()
	config.gpu_options.allow_growth = True
	sess = Session(config=config)
else:
	from tensorflow import config

	physical_devices = config.experimental.list_physical_devices('GPU')
	config.experimental.set_memory_growth(physical_devices[0], True)
#from tensorflow.keras import backend as K

# Constants
CUBOIDS_LENGTH = 8
CUBOIDS_WIDTH = 224
CUBOIDS_HEIGHT = 224

# Image resize function
resize_fn = lambda img: np.expand_dims(resize(cvtColor(img, COLOR_BGR2GRAY),
						(CUBOIDS_WIDTH, CUBOIDS_HEIGHT))/255, axis=2)

### Input Arguments
parser = argparse.ArgumentParser(description='Test an Incremental Spatio'\
							' Temporal Learner model for a given test dataset')
parser.add_argument('-m', '--model', help='A pretrained model stored on a'\
					' h5 file', type=str)
parser.add_argument('-c', '--train_folder', help='Path to folder'\
					' containing the train dataset', type=str, nargs='?')
parser.add_argument('-d', '--data_folder', help='Path to folder'\
					' containing the test dataset', type=str)
parser.add_argument('-l', '--labels', help='Path to file containing the test'\
					' labels', type=str)
parser.add_argument('-a', '--anom_threshold',
					help='Anomaly threshold values to test', type=float,
					nargs=1)
parser.add_argument('-t', '--temp_threshold',
					help='Temporal threshold values to test', type=int,
					nargs=1)
parser.add_argument('-o', '--output', help='Output file in which the results'\
					' will be located', type=str)

args = parser.parse_args()

model_fn = args.model
train_video_dir = args.train_folder
test_video_dir = args.data_folder
labels_path = args.labels
anom_threshold = args.anom_threshold[0]
temp_threshold = args.temp_threshold[0]
output = args.output

"""
dot_pos = output.rfind('.')
if dot_pos != -1:
	results_fn = output[:dot_pos] + '.json'
else:
	results_fn = output + '.json'
"""

### Create output directory
if not os.path.isdir(output):
	try:
		os.mkdir(output)
	except Exception as e:
		print('Failed to create output directory {}'.format(str(e)),
				file=sys.stderr)
		exit(-1)


### Loads model
try:
	model = load_model(model_fn, custom_objects={'root_sum_squared_error':                   
							root_sum_squared_error})
except Exception as e:
	print('Cannot load the model: ', str(e), file=sys.stderr)
	exit(-1)


### Load the video test dataset
if train_video_dir:
	try:
		data_train = istl.generators.CuboidsGeneratorFromImgs(
										source=train_video_dir,
										cub_frames=CUBOIDS_LENGTH,
										prep_fn=resize_fn)
	except Exception as e:
		print('Cannot load {}: '.format(train_video_dir), str(e), file=sys.stderr)
		exit(-1)
else:
	data_train = None

try:
	data_test = istl.generators.CuboidsGeneratorFromImgs(source=test_video_dir,
									cub_frames=CUBOIDS_LENGTH,
									prep_fn=resize_fn)
	data_test = istl.generators.ConsecutiveCuboidsGen(data_test)
except Exception as e:
	print('Cannot load {}: '.format(test_video_dir), str(e), file=sys.stderr)
	exit(-1)

# Note the cumulative cuboids per each test video
cum_cuboids_per_video = data_test.cum_cuboids_per_video

try:
	test_labels = np.loadtxt(labels_path, dtype='int8')
except Exception as e:
	print('Cannot load {}: '.format(labels_path), str(e), file=sys.stderr)
	exit(-1)

### Testing for each pair of anomaly and temporal values combinations
print('Performing evaluation with the anomaly and temporal thesholds')
evaluator = istl.EvaluatorISTL(model=model,
									cub_frames=CUBOIDS_LENGTH,
									anom_thresh=anom_threshold,
									temp_thresh=temp_threshold)

if data_train is not None:
	sc_train = evaluator.fit(data_train)

try:
	scale = cum_cuboids_per_video if data_train is None else None


	pred, scores_test, true_scores_test = evaluator.predict_cuboids(
								cub_set=data_test,
								return_scores=True,
								cum_cuboids_per_video=scale)
except Exception as e:
	print(str(e))
	exit(-1)

meas = {}
if data_train is not None:
	meas['training_rec_error'] = {
								'mean': sc_train.mean(),
								'std': sc_train.std(),
								'min': sc_train.min(),
								'max': sc_train.max()
								}

meas['test_rec_error'] = {
						'mean': true_scores_test.mean(),
						'std': true_scores_test.std(),
						'min': true_scores_test.min(),
						'max': true_scores_test.max()
						}

meas['test_rec_error_norm'] = {
						'mean': scores_test.mean(),
						'std': scores_test.std(),
						'min': scores_test.min(),
						'max': scores_test.max()
						}

meas['results'] = istl.EvaluatorISTL._compute_perf_metrics(
											test_labels, pred,
											true_scores_test)

###### Separate prediction, scores and true labels per video ######
scores_test_sep = split_measures_per_video(scores_test, cum_cuboids_per_video)
pred_sep = split_measures_per_video(pred, cum_cuboids_per_video)
test_labels_sep = split_measures_per_video(test_labels, cum_cuboids_per_video)

for i in range(len(scores_test_sep)):

	plt_fname = os.path.join(output, 'test{}'.format(i + 1))

	# Indexes of predictions
	idxs = np.arange(pred_sep[i].size)

	# Compare ground-truth with true labels
	comp = np.zeros(pred_sep[i].size, dtype='uint8')

	#comp[pred_sep==0 && test_labels_sep==0] = 0 # TN
	comp[(pred_sep[i]==1) & (test_labels_sep[i]==1)] = 1 # TP
	comp[(pred_sep[i]==0) & (test_labels_sep[i]==1)] = 2 # FN
	comp[(pred_sep[i]==1) & (test_labels_sep[i]==0)] = 3 # FP

	# Graph true labels and predictions
	#fig, ax = plt.subplots(ncols=3)

	# 	True labels
	plt.plot(idxs[test_labels_sep[i]==0], scores_test_sep[i][test_labels_sep[i]==0], '.', label='Normal')
	plt.plot(idxs[test_labels_sep[i]==1], scores_test_sep[i][test_labels_sep[i]==1], '.', label='Abnormal')
	plt.plot(idxs, np.repeat(anom_threshold, repeats=len(idxs)), '--', label='Anom threshold')
	plt.legend()
	plt.title('Ground Truth')
	plt.xlabel('frames')
	plt.ylabel('Rec Error Norm')

	plt.savefig(plt_fname+'_ground_truth.svg')
	plt.close()

	# 	Predictions
	plt.plot(idxs[pred_sep[i]==0], scores_test_sep[i][pred_sep[i]==0], '.', label='Normal')
	plt.plot(idxs[pred_sep[i]==1], scores_test_sep[i][pred_sep[i]==1], '.', label='Abnormal')
	plt.plot(idxs, np.repeat(anom_threshold, repeats=len(idxs)), '--', label='Anom threshold')
	plt.legend()
	plt.title('Prediction')
	plt.xlabel('frames')
	plt.ylabel('Rec Error Norm')

	plt.savefig(plt_fname+'_prediction.svg')
	plt.close()

	#	Comparation
	plt.plot(idxs[comp[i]==0], scores_test_sep[i][comp[i]==0], '.', label='True Positive (TP)')
	plt.plot(idxs[comp[i]==1], scores_test_sep[i][comp[i]==1], '.', label='True Negative (TN)')
	plt.plot(idxs[comp[i]==2], scores_test_sep[i][comp[i]==2], '.', label='False Negative (FN)')
	plt.plot(idxs[comp[i]==3], scores_test_sep[i][comp[i]==3], '.', label='False Positive (FP)')
	plt.plot(idxs, np.repeat(anom_threshold, repeats=len(idxs)), '--', label='Anom threshold')
	plt.legend()
	plt.title('Comparation')
	plt.xlabel('frames')
	plt.ylabel('Rec Error Norm')

	plt.savefig(plt_fname+'_comparation.svg')
	plt.close()

# Save the results
with open(os.path.join(output, 'measures.json'), 'w') as f:
	json.dump(meas, f, indent=4)
