# -*- coding: utf-8 -*-
"""

@author: Nicolás Cubero Torres
@description: Trains and evaluate an Incremental Spatio Temporal Learner (ISTL)
				model by using the UCSD Ped 1 train and test sets on a
				federated architecture simulated on a single node.

			This scripts replicates the experiment carried out on [1] for the
			UCSD Ped 1 Dataset adding the federated learning architecture in
			which the experiment is carried out:

			In this case, active learning is not applied, so training data will
			not be partitioned and training will be performed on offline.

@usage: train_ISTL_UCSDPed1_noAct.py -d <JSON document experiment> [-s]
"""

# Modules imported
import time
import sys
import argparse
import json
import numpy as np
from tensorflow.keras.models import clone_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow import config
from cv2 import resize, cvtColor, COLOR_BGR2GRAY
from utils import extract_experiments_parameters, plot_results
from fedLearn import SynFedAvgLearnModel
from models import istl

# Constants
CUBOIDS_LENGTH = 8
CUBOIDS_WIDTH = 224
CUBOIDS_HEIGHT = 224

# Image resize function
resize_fn = lambda img: np.expand_dims(resize(cvtColor(img, COLOR_BGR2GRAY),
						(CUBOIDS_WIDTH, CUBOIDS_HEIGHT))/255, axis=2)


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

exp_data['script'] = __file__

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

data_train = istl.generators.CuboidsGeneratorFromImgs(
		source=train_video_dir,
		cub_frames=CUBOIDS_LENGTH,
		prep_fn=resize_fn)

data_test = istl.generators.CuboidsGeneratorFromImgs(source=test_video_dir,
									cub_frames=CUBOIDS_LENGTH,
									prep_fn=resize_fn)
data_test = istl.generators.ConsecutiveCuboidsGen(data_test)
test_labels = np.loadtxt(test_label, dtype='int8')

## Configure GPU usage
physical_devices = config.experimental.list_physical_devices('GPU')
config.experimental.set_memory_growth(physical_devices[0], True)

# Perform training for each parameters combination
results = []
params = extract_experiments_parameters(exp_data, ('seed', 'batch_size'))

for p in params:

	if 'seed' in p:
		np.random.seed(p['seed'])

	# Prepare the data train and make partitions
	#data_train.shuffle(shuf=bool(p['shuffle']) if 'shuffle' in p else False,
	#						seed=p['seed'] if 'seed' in p else time.time())

	# The generators must return the cuboids batch as label also when indexing
	data_train.return_cub_as_label = True
	data_train.batch_size = p['batch_size'] if 'batch_size' in p else 1

	# Split data for each client
	train_split = data_train.make_partitions((0.5, 0.5))

	# Augment the cuboids corresponding to the first partition
	for split in train_split:
		split.augment_data(max_stride=3)

	t_start = time.time()

	print('Training with parameters: {}'.format(p))

	#################    Model preparation    ################
	data = {0: train_split[0],
			1: train_split[1]}

	# Stochastic gradient descent algorithm
	adam = Adam(lr=1e-4, decay=5e-3, epsilon=1e-6)


	istl_fed_model = SynFedAvgLearnModel(build_fn=istl.build_ISTL, n_clients=2,
										cub_length=CUBOIDS_LENGTH)
	istl_fed_model.compile(optimizer=adam, loss=MeanSquaredError())


	########## Training  ##########
	t_1it_start = time.time()
	print('Training')
	#print('- {} samples'.format(len(data_train)))

	hist = istl_fed_model.fit(x=data,
						epochs=p['epochs'],
						early_stop_monitor='loss',
						early_stop_patience=p['early_stop_patience'] if 'early_stop_patience' in p else 5,
						early_stop_delta=p['early_stop_delta'] if 'early_stop_delta' in p else 1e-6,
						early_stop_rest_best_weights = True,
						backup_filename='backup.h5',
						backup_epochs=10,
						backup_save_only_weights=False,
						verbose=2,
						shuffle=False)

	t_1it_end = time.time()
	p['time'] = {'Training': (t_1it_end - t_1it_start)}
	print('End of training - elapsed time {} s'.format(p['time']
															['Training']))

	# Plot MSE
	for c in range(2):
		plot_results({'MSE - training': hist[c]['loss']},
			'Mean Squared Error',
			model_base_filename +
			'ISTL_UCSDPed1_MSE_train_loss_client={}_exp={}.pdf'.format(c, len(results)+1))

		np.savetxt(model_base_filename +
			'ISTL_UCSDPed1_MSE_train_loss_client={}_exp={}.txt'.format(c, len(results)+1),
					hist[c]['loss'])

	## Save model
	if store_models:
		istl_fed_model.global_model.save(model_base_filename +
							'-experiment-'+str(len(results)) + '_model.h5')

	########### Test ##############
	t_eval_start = time.time()
	evaluator = istl.EvaluatorISTL(model=istl_fed_model.global_model,
										cub_frames=CUBOIDS_LENGTH,
										# It's required to put any value
										anom_thresh=0.1,
										temp_thresh=1)

	data_train.return_cub_as_label = False
	data_train.batch_size = 1
	data_train.shuffle(False)
	evaluator.fit(data_train)

	t_eval_end = time.time()
	p['time']['test evaluation'] = (t_eval_end - t_eval_start)

	print('Performing evaluation with all anomaly and temporal '\
			'thesholds combinations')
	all_meas = evaluator.evaluate_cuboids_range_params(data_test,
											test_labels,
											np.arange(0.01, 1, 0.01),
											np.arange(1,10))
	p['results']= {'test all combinations': all_meas}

	p['time']['total_elapsed time'] = (p['time']['test evaluation'] +
											p['time']['Training'])
	print('End of experiment - Total time taken: {}s'.format(p['time']
													['total_elapsed time']))

	results.append(p)

	# Save the results
	with open(results_filename, 'w') as f:
		json.dump(results, f, indent=4)
