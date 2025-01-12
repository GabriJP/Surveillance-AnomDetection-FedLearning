# -*- coding: utf-8 -*-
"""

@author: Nicolás Cubero Torres
@description: Trains and evaluate an Incremental Spatio Temporal Learner (ISTL)
	model by using the UCSD Ped 1 and UCSD Ped 2 train and test sets.

	This scripts replicates the experiment carried out on [1] for the UCSD Ped 1
	and UCSD Ped 2 datasets without applying federated learning nor active
	learning, so training data will not be partitioned and training will be
	performed on offline.

	The script receives experiment description from a JSON document and saves
	the results of each experiment on separate JSON document located at the
	same directory as the description join to the h5 models (if model saving
	flag is specified)

	The document experiment must contains the following fields:

	Mandatory:

	  "train_video_dir": (str)
	  		Directory path containing the train set

	  "test_video_dir": (str)
	  		Directory path containing the test set

	  "test_label": (str)
	  		Filepath locating the test cuboids labels (as txt format).

	  "batch_size": (int)|(list of int)
	  		Values to be used as batch size for each experiment.

	  "max_stride": (int), default: 3
	  		Maximum stride applied for train set augmentation

	  "anom_thresh": (float in [0,1])|(list of floats)
	  		Anomaly thresholds to be used on each experiment

	  "temp_thresh": (float in [0,1])|(list of ints)
	  		Temporal thresholds to be used on each experiment

	  Optional:

	  "epochs": (int), default: 1
	  		Max number of epochs performed for the training on each iteration

	  "seed": (int)
	  		Value used as seed for the generator.

	  "lr": (float), default: 1e-4
	  		Initial learning rate used for training

	  "port_val": (float in (0, 1]), default: 0.1
	  		Ration of train samples for each client and iteration to be used for
			validation.

	  "shuffle": (int containing 1 or 0)
	  		Whether shuffle train samples (1) or not (0).

	  "patience": (int), default: 10.
	  		Max number of epochs with no improvement on validation loss before
			reducing learning rate on lr_decay factor.

	NOTE: For those parameter for which a list of values are provided, an
	an experiment for each combination is performed

	For example, on providing "batch_size": [20, 32] and "seed": [20, 100],
	four experiment will be performed for each possible combination of batch
	size single value and seed single value

@usage: train_ISTL_noFed_noAct_UCSDPed_val.py -d <JSON document experiment>
											[-s] save the resulting model
"""

# Modules imported
import time
import sys
import argparse
import json
import numpy as np
import tensorflow
from tensorflow.keras.models import clone_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow import config, random
from cv2 import resize, cvtColor, COLOR_BGR2GRAY
from utils import extract_experiments_parameters, plot_results, root_sum_squared_error
from fedLearn import SynFedAvgLearnModel
from models import istl
from learningRateImprover import LearningRateImprover

# Constants
CUBOIDS_LENGTH = 8
CUBOIDS_WIDTH = 224
CUBOIDS_HEIGHT = 224

# Image resize function
resize_fn = lambda img: np.expand_dims(resize(cvtColor(img, COLOR_BGR2GRAY),
						(CUBOIDS_WIDTH, CUBOIDS_HEIGHT))/255, axis=2)

### Input Arguments
parser = argparse.ArgumentParser(description='Trains an Incremental Spatio'\
							' Temporal Learner model for the UCSD Ped 1 or'\
							'UCSD Ped 2 datasets.')
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
	results_filename = exp_filename[:dot_pos] + '_experimento-{}.json'
	model_base_filename = exp_filename[:dot_pos]
else:
	results_filename = exp_filename + '_experimento-{}.json'
	model_base_filename = exp_filename[:]

### Data loading and preparation ###
train_video_dir = exp_data['train_video_dir']
test_video_dir = exp_data['test_video_dir']
test_label = exp_data['test_label']

data_test = istl.generators.CuboidsGeneratorFromImgs(source=test_video_dir,
									cub_frames=CUBOIDS_LENGTH,
									prep_fn=resize_fn,
									max_cuboids=10000)
data_test = istl.generators.ConsecutiveCuboidsGen(data_test)
test_labels = np.loadtxt(test_label, dtype='int8')

## Configure GPU usage
physical_devices = config.experimental.list_physical_devices('GPU')
config.experimental.set_memory_growth(physical_devices[0], True)

# Perform training for each parameters combination
results = []
params = extract_experiments_parameters(exp_data, ('seed', 'batch_size',
												'lr_decay', 'max_stride'))

for p in params:

	try:
		if 'seed' in p:
			np.random.seed(p['seed'])

			if tensorflow.__version__.startswith('1'):
				random.set_random_seed(p['seed'])
			else:
				random.set_seed(p['seed'])

		# Prepare the data train and make partitions
		#data_train.shuffle(shuf=bool(p['shuffle']) if 'shuffle' in p else False,
		#						seed=p['seed'] if 'seed' in p else time.time())

		# The generators must return the cuboids batch as label also when indexing
		data_train = istl.generators.CuboidsGeneratorFromImgs(
													source=train_video_dir,
													cub_frames=CUBOIDS_LENGTH,
													prep_fn=resize_fn,
													max_cuboids=10000)
		data_train.return_cub_as_label = True
		data_train.batch_size = p['batch_size'] if 'batch_size' in p else 1
		data_val, data_train = data_train.take_subpartition(
									p['port_val'] if 'port_val' in p else 0.1,
									p['seed'] if 'seed' in p else None)

		# Augment the cuboids
		data_train.augment_data(max_stride=p['max_stride'] if 'max_stride' in p else 3)
		data_train.shuffle(shuf=bool(p['shuffle']) if 'shuffle' in p else False,
									seed=p['seed'] if 'seed' in p else time.time())

		t_start = time.time()

		print('Training with parameters: {}'.format(p))

		#################    Model preparation    ################

		# Stochastic gradient descent algorithm
		adam = Adam(lr=1e-4, decay=p['lr_decay'] if 'lr_decay' in p else 0,
					epsilon=1e-6)

		istl_model = istl.build_ISTL(cub_length=CUBOIDS_LENGTH)
		istl_model.compile(optimizer=adam, loss=MeanSquaredError(),
							metrics=[root_sum_squared_error])


		########## Training  ##########
		t_1it_start = time.time()
		print('Training')
		#print('- {} samples'.format(len(data_train)))

		patience = p['patience'] if 'patience' in p else 0
		epochs = p['epochs'] if 'epochs' in p else 1
		callbacks = [LearningRateImprover(parameter='val_loss',
										min_lr=1e-8, factor=0.9, patience=patience,
										min_delta=1e-6, verbose=1,
										restore_best_weights=True),
									ModelCheckpoint(filepath='backup.h5',
														monitor='val_loss',
														save_freq='epoch',
														verbose=1)]
		hist = istl_model.fit(x=data_train, epochs=epochs,
							validation_data=data_val,
							callbacks=callbacks,
							verbose=2,
							shuffle=False)
		# 										ModelCheckpoint(filepath='backup.h5',
			#											monitor='loss',
			#											save_freq='epoch',
			#											verbose=1)

		hist_val_rec = hist.history['val_root_sum_squared_error']
		hist_val = hist.history['val_loss']
		hist_rec = hist.history['root_sum_squared_error']
		hist = hist.history['loss']

		t_1it_end = time.time()
		p['time'] = {'Training': (t_1it_end - t_1it_start)}
		print('End of training - elapsed time {} s'.format(p['time']
																['Training']))

		# Plot MSE
		plot_results({'MSE - Training': hist, 'MSE - Validation': hist_val},
			'Mean Squared Error',
			model_base_filename +
			'ISTL_MSE_train_loss_exp={}.pdf'.format(len(results)+1))

		np.savetxt(model_base_filename +
			'ISTL_MSE_train_loss_exp={}.txt'.format(len(results)+1),
					hist)

		np.savetxt(model_base_filename +
			'ISTL_MSE_val_loss_exp={}.txt'.format(len(results)+1),
					hist_val)

		# Plot RSSE
		plot_results({'RSSE - Training': hist_rec,
						'RSSE - Validation': hist_val_rec},
			'Root of the Sum of Squared Errors',
			model_base_filename +
			'ISTL_RSSE_train_loss_exp={}.pdf'.format(len(results)+1))

		np.savetxt(model_base_filename +
			'ISTL_RSSE_train_loss_exp={}.txt'.format(len(results)+1),
					hist_rec)

		np.savetxt(model_base_filename +
			'ISTL_RSSE_validation_loss_exp={}.txt'.format(len(results)+1),
					hist_val_rec)

		# Plot lr history
		plot_results({'Lr history': callbacks[0].lr_history},
			'Learning rate history',
			model_base_filename +
			'ISTL_lr_history_exp={}.pdf'.format(len(results)+1))

		np.savetxt(model_base_filename +
			'ISTL_lr_history_exp={}.txt'.format(len(results)+1),
					callbacks[0].lr_history)

		## Save model
		if store_models:
			istl_model.save(model_base_filename +
								'-experiment-'+str(len(results)) + '_model.h5')

		########### Test ##############
		t_eval_start = time.time()
		evaluator = istl.EvaluatorISTL(model=istl_model,
											cub_frames=CUBOIDS_LENGTH,
											# It's required to put any value
											anom_thresh=0.1,
											temp_thresh=1)

		data_train.return_cub_as_label = False
		data_train.batch_size = 1
		data_train.shuffle(False)
		train_rec_error = evaluator.score_cuboids(data_train, False)

		p['training_rec_errors'] = {
									'mean': train_rec_error.mean(),
									'std': train_rec_error.std(),
									'min': train_rec_error.min(),
									'max': train_rec_error.max()
								}

		t_eval_end = time.time()
		p['time']['test evaluation'] = (t_eval_end - t_eval_start)

		print('Performing evaluation with all anomaly and temporal '\
				'thesholds combinations')
		all_meas = evaluator.evaluate_cuboids_range_params(data_test,
												test_labels,
												np.arange(0.01, 1, 0.01),
												np.arange(1, 10),
												data_test.cum_cuboids_per_video)
		p['results']= {'test all combinations': all_meas}

		p['time']['total_elapsed time'] = (p['time']['test evaluation'] +
												p['time']['Training'])
		print('End of experiment - Total time taken: {}s'.format(p['time']
														['total_elapsed time']))

	except Exception as e:
		p['results'] = 'Failed to execute the experiment: ' + str(e)
		raise

	results.append(p)

	# Save the results
	with open(results_filename.format(len(results)), 'w') as f:
		json.dump(p, f, indent=4)
