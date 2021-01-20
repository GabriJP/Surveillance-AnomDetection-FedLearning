# -*- coding: utf-8 -*-
"""

@author: Nicolás Cubero Torres
@description: Evaluate the prediction of an ISTL model for a given test dataset.

@usage: evaluate_ISTL.py -m <Pretrained h5 model file>
					-d <Directory Path containing the test set to evaluate>
					-l <File containing the test labels>
					-a <Anomaly threshold values to evaluate>
					-t <Temporal threshold values to evaluate>
					-o <Output directory>
					[-c <Directory Path containing the train set used for
						fitting the reconstruction error scaling>]
					[-s] Perform spatial location over the test samples
					[-n] Sets normalization between 0 and 1 when escaling the
						reconstruction error over the test reconstruction error
						(not -c option used).
					[--train_cons_cuboids] Use consecutive cuboids extraction
						for training set
"""
# Modules imported
import sys
import json
import argparse
import numpy as np
from cv2 import imread, imwrite, resize, cvtColor, COLOR_BGR2GRAY
import tensorflow
from tensorflow.keras.models import load_model
from models import istl
from utils import root_sum_squared_error

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
					nargs='+')
parser.add_argument('-t', '--temp_threshold',
					help='Temporal threshold values to test', type=int,
					nargs='+')
parser.add_argument('-o', '--output', help='Output file in which the results'\
					' will be located', type=str)
parser.add_argument('-n', '--norm_zero_one', help='Set the normalization between'\
					' zero or one. Only valid for the evaluation per test'\
					' sample', action='store_true')
parser.add_argument('--train_cons_cuboids', help='Set the usage of overlapping'\
					' cuboids for the training set', action='store_true')

args = parser.parse_args()

model_fn = args.model
train_video_dir = args.train_folder
test_video_dir = args.data_folder
labels_path = args.labels
anom_threshold = args.anom_threshold
temp_threshold = args.temp_threshold
output = args.output
norm_zero_one = args.norm_zero_one
train_cons_cuboids = args.train_cons_cuboids

dot_pos = output.rfind('.')
if dot_pos != -1:
	results_fn = output[:dot_pos] + '.json'
else:
	results_fn = output + '.json'


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

		# Use overlapping cuboids for training set
		if train_cons_cuboids:
			data_train = istl.generators.ConsecutiveCuboidsGen(data_train)

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

try:
	test_labels = np.loadtxt(labels_path, dtype='int8')
except Exception as e:
	print('Cannot load {}: '.format(labels_path), str(e), file=sys.stderr)
	exit(-1)

### Testing for each pair of anomaly and temporal values combinations
print('Performing evaluation with all anomaly and temporal '\
		'thesholds combinations')
evaluator = istl.EvaluatorISTL(model=model,
									cub_frames=CUBOIDS_LENGTH,
									# It's required to put any value
									anom_thresh=0.1,
									temp_thresh=1)

if data_train is not None:
	sc_train = evaluator.fit(data_train)

try:
	scale = data_test.cum_cuboids_per_video if data_train is None else None

	all_meas = evaluator.evaluate_cuboids_range_params(data_test,
											test_labels,
											anom_threshold,
											temp_threshold,
											scale,
											norm_zero_one)
except Exception as e:
	print(str(e))
	exit(-1)

if data_train is not None:
	all_meas['training_rec_error'] = {
									'mean': float(sc_train.mean()),
									'std': float(sc_train.std()),
									'min': float(sc_train.min()),
									'max': float(sc_train.max())
									}

# Save the results
with open(results_fn, 'w') as f:
	json.dump(all_meas, f, indent=4)
