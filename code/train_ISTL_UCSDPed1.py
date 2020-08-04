# -*- coding: utf-8 -*-
"""

@author: Nicolás Cubero Torres
@description: Trains and evaluate an Incremental Spatio Temporal Learner (ISTL)
				model by using the UCSD Ped 1 train and test sets on a
				federated architecture simulated on a single node.

			This scripts replicates the experiment carried out on [1] for the
			UCSD Ped 1 Dataset adding the federated learning architecture in
			which the experiment is carried out:

			The UCSD Ped 1 train dataset is splitted in three partitions: the
			first with the 60 % of data, and 20 % for each second and third
			partition, then the first partition is trained in a offline method
			and the two remainders partitions are trained with active learning.

@usage: train_ISTL_UCSDPed1.py -d <JSON document experiment> [-s]
"""

# Modules imported
import time
import sys
import argparse
import numpy as np
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping
from cv2 import resize, cvtColor, CV_BGR2GRAY
from utils import extract_experiments_parameters, plot_results
from fedLearn import SynFedAvgLearnModel
from models import istl

# Constants
CUBOIDS_LENGTH = 8
CUBOIDS_WIDTH = 224
CUBOIDS_HEIGHT = 224

# Image resize function
resize_fn = lambda img: resize(cvtColor(img, CV_BGR2GRAY),
						(CUBOIDS_WIDTH, CUBOIDS_HEIGHT))/255

### Input Arguments
parser = argparse.ArgumentParser(description='Trains an Incremental Spatio'\
							' Temporal Learner model for the UCSD Ped 1'\
							'dataset by using active learning on a federated '\
							'architecture')
parser.add_argument('-d', '--document', help='JSON file containing the train'\
					' parameters', type=str)
parser.add_argument('-s', '--save_model', help='Save the resulting model'\
					' on a h5 file',
					action='store_true', default=False)

args = parser.parse_args()

exp_filename = args.document
store_models = args.save_model

# Read experiment document
with open(exp_filename) as f:
	try:
		exp_data = json.load(f)
	except Exception as e:
		print('Cannot load experiment JSON file'\
			' :\n',str(e), file=sys.stderr)
		exit(-1)

# Get output filenames
dot_pos = exp_filename.rfind('.')
if dot_pos != -1:
	results_filename = exp_filename[:dot_pos] + '_experimentos.json'
	model_base_filename = exp_filename[:dot_pos]
else:
	results_filename = exp_filename + '_experimentos.json'
	model_base_filename = exp_filename[:]

### Data loading and preparation ###
train_video_dir = exp_data['train_video_dir']
test_video_dir = exp_data['test_video_dir']
test_label = exp_data['test_label']

data_train = istl.CuboidsGenerator(source=train_video_dir,
									cub_frames=CUBOIDS_LENGTH,
									prep_function=resize_fn,
									shuffle=exp_data['shuffle'])

data_test = istl.CuboidsGenerator(source=test_video_dir,
									cub_frames=CUBOIDS_LENGTH,
									prep_function=resize_fn)
test_labels = np.loadtxt(test_label, dtype='int8')

train_split = data_train.make_partitions((0.3, 0.3, 0.1, 0.1, 0.1, 0.1))

# Perform training for each parameters combination
results = []
params = extract_experiments_parameters(exp_data, ('seed', 'anom_thresh',
																'temp_thresh'))

for p in params:
	np.random.seed(p['seed'])

	t_start = time.time()

	print('Training with parameters: {}'.format(p))

	### Model preparation ###
	sgd = SGD(learning_rate=1e-2)		# Stochastic gradient descent algorithm

	istl_fed_model = SynFedAvgLearnModel(build_fn=istl.build_ISTL(), n_clients=2)
	istl_fed_model.compile(optimizer=sgd, loss='mean_squared_error')

	### Training ###

	# First iteration (training with the 60% of data)
	data = {0: istl.FramesFromCuboidsGen(train_split[0]),
			1: istl.FramesFromCuboidsGen(train_split[1])}

	hist1 = istl_fed_model.fit(x=data, y=data, epochs=p['epochs'],
						early_stop_monitor='mean_squared_error',
						early_stop_patience=p['early_stop_patience'] if 'early_stop_patience' in p else 5,
						early_stop_delta=p['early_stop_delta'] if 'early_stop_delta' in p else 1e-7,
						verbose=2)

	# Plot MSE of client training
	for c in range(2):
		plot_results({'MSE - 1st iteration - client #{}'.format(c): hist1[c]['loss']},
			'Mean Squared Error',
			model_base_filename +
			'ISTL_UCSDPed1_1st_iteration_client{}_MSE_train_loss.pdf'.format(c))

	# Second iteration (training with the 20% of data)
	data = {0: train_split[2], 1: train_split[3]}

	# Evaluate performance and retrieve false positive cuboids
	# to train with them
	evaluator = istl.EvaluatorLSTM(model=istl_fed_model.global_model,
									cub_frames=CUBOIDS_LENGTH,
									anom_thresh=p['anom_thresh'],
									temp_thresh=p['temp_thresh'])

	p['results'] = {}
	p['results']['2nd iteration'] = {}

	for c in data:
		evaluator.clear()

		meas = evaluator.evaluate_cuboids(data[c], [0]*len(data[c]))
		print('Evaluation of second iteration client set {} founding {} false positive'\
				' cuboids\n{}'.format(c, len(evaluator), meas))

		p['results']['2nd iteration'][c] = {'fp cuboids': len(evaluator),
												'measures': meas}

		data[c] = istl.FramesFromCuboidsGen(evaluator.fp_cuboids) if len(evaluator) else None

	# Save the results
	results.append(p)

	with open(results_filename, 'w') as f:
		json.dump(results, f, indent=4)

	hist2 = istl_fed_model.fit(x=data, y=data, epochs=p['epochs'],
						early_stop_monitor='mean_squared_error',
						early_stop_patience=p['early_stop_patience'] if 'early_stop_patience' in p else 5,
						early_stop_delta=p['early_stop_delta'] if 'early_stop_delta' in p else 1e-7,
						verbose=2)

	# Plot MSE of client training
	for c in range(2):
		plot_results({'MSE - 2nd iteration - client #{}'.format(c): hist2[c]['loss']},
			'Mean Squared Error',
			model_base_filename+
			'ISTL_UCSDPed1_2nd_iteration_client{}_MSE_train_loss.pdf'.format(c))

	# Third iteration (training with the 20% of data)
	data = {0: train_split[4], 1: train_split[5]}

	# Evaluate performance and retrieve false positive cuboids
	# to train with them
	p['results']['3rd iteration'] = {}

	for c in data:
		evaluator.clear()

		meas = evaluator.evaluate_cuboids(data[c], [0]*len(data[c]))
		print('Evaluation of third iteration client set {} founding {} false positive'\
				' cuboids\n{}'.format(c, len(evaluator), meas))

		p['results']['3rd iteration'][c] = {'fp cuboids': len(evaluator),
												'measures': meas}

		data[c] = istl.FramesFromCuboidsGen(evaluator.fp_cuboids) if len(evaluator) else None

	# Save the results (p is yet stored in the list)
	with open(results_filename, 'w') as f:
		json.dump(results, f, indent=4)

	hist3 = istl_fed_model.fit(x=data, y=data, epochs=p['epochs'],
						early_stop_monitor='mean_squared_error',
						early_stop_patience=p['early_stop_patience'] if 'early_stop_patience' in p else 5,
						early_stop_delta=p['early_stop_delta'] if 'early_stop_delta' in p else 1e-7,
						verbose=2)

	# Plot MSE of client training
	for c in range(2):
		plot_results({'MSE - 3rd iteration - client #{}'.format(c): hist3[c]['loss']},
			'Mean Squared Error',
			model_base_filename +
			'ISTL_UCSDPed1_3rd_iteration_client{}_MSE_train_loss.pdf'.format(c))


	## Save model
	if store_models:
		istl_fed_model.global_model.save(model_base_filename +
							'-experiment-'+str(len(results)+1) + '_model.h5')

	## Final evaluation
	evaluator.clear()
	meas = evaluator.evaluate_cuboids(data_test, test_labels)

	p['results']['test set'] = {'fp cuboids': len(evaluator),
									'measures': meas}

	print('Evaluation of test set founding {} false positive'\
				' cuboids\n{}'.format(len(evaluator), meas))

	t_end = time.time()
	p['elapsed time'] = float(t_end - t_start)
	print('Time taken: {}s'.format(p['elapsed time']))
