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

@usage: train_ISTL_UCSDPed1_val.py -d <JSON document experiment> [-s]
"""

# Modules imported
import time
import sys
import argparse
import json
from copy import deepcopy
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
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
	train_split = data_train.make_partitions(p['partitions'])

	t_start = time.time()

	print('Training with parameters: {}'.format(p))

	#################    Model preparation    ################
	# Stochastic gradient descent algorithm
	#sgd = SGD(learning_rate=p['lr'] if 'lr' in p else 1e-2)
	adam = Adam(lr=1e-4, decay=1e-5, epsilon=1e-6)

	istl_fed_model = SynFedAvgLearnModel(build_fn=istl.build_ISTL, n_clients=2,
										cub_length=CUBOIDS_LENGTH)
	istl_fed_model.compile(optimizer=adam, loss='mean_squared_error')




	########## First iteration (training with the 60% of data) ##########
	t_1it_start = time.time()
	print('Training on 1st iteration')

	data = {0: train_split[0], 1: train_split[1]}
	val_data = {}
	for c in data:
		data[c].batch_size = p['batch_size'] if 'batch_size' in p else 1

		# The generators must return the cuboids batch as label also
		# when indexing
		data[c].return_cub_as_label = True

		# Set validation data
		data[c], val_data[c] = data[c].make_partitions((0.9, 0.1))

		# Set data augmentation
		data[c].augment_data(max_stride=3)
		data[c].shuffle(shuf=bool(p['shuffle']) if 'shuffle' in p else False,
								seed=p['seed'] if 'seed' in p else time.time())

	print('- Client 0: {} samples, Client 1: {} samples'.format(
				len(data[0]),
				len(data[1])))

	hist1 = istl_fed_model.fit(x=data, epochs=p['epochs'],
						validation_data=val_data,
						early_stop_monitor='val_loss',
						early_stop_patience=p['early_stop_patience'] if 'early_stop_patience' in p else 5,
						early_stop_delta=p['early_stop_delta'] if 'early_stop_delta' in p else 1e-6,
						backup_filename='backup.h5',
						backup_epochs=10,
						backup_save_only_weights=False,
						verbose=2,
						shuffle=False)
	hist1_val = {c: hist1[c]['val_loss'] for c in hist1}
	hist1 = {c: hist1[c]['loss'] for c in hist1}

	t_1it_end = time.time()
	p['time'] = {'1st iteration': (t_1it_end - t_1it_start)}
	print('End of 1st iteration - elapsed time {} s'.format(p['time']
															['1st iteration']))

	# Plot MSE of client training
	for c in range(2):
		plot_results({'MSE - training': hist1[c],
						'MSE - validation': hist1_val[c]},
			'Mean Squared Error - 1st iteration - client #{}'.format(c),
			model_base_filename +
			'ISTL_1st_iteration_client{}_MSE_train_loss_exp={}.pdf'.format(c, len(results)+1))

		np.savetxt(model_base_filename +
			'ISTL_1st_iteration_client{}_MSE_loss_exp={}.txt'.format(c, len(results)+1),
					hist1[c])

		np.savetxt(model_base_filename +
			'ISTL_1st_iteration_client{}_MSE_val_loss_exp={}.txt'.format(c, len(results)+1),
					hist1_val[c])

	## Perform the second and third iteration for each combination of anomaly
	## thresholds and temporal thresholds
	thresh_params = extract_experiments_parameters(p, ('anom_thresh',
														'temp_thresh',
														'force_relearning'))

	for q in thresh_params:

		istl_fed_model_copy = deepcopy(istl_fed_model)
		q['#experiment'] = len(results) + 1
		q['force_relearning'] = (bool(q['force_relearning'])
										if 'force_relearning' in q else False)

		########## Second iteration (training with the 20% of data) ###########

		# Evaluate performance and retrieve false positive cuboids
		# to train with them
		evaluator = istl.EvaluatorISTL(model=istl_fed_model_copy.global_model,
										cub_frames=CUBOIDS_LENGTH,
										anom_thresh=q['anom_thresh'],
										temp_thresh=q['temp_thresh'])

		evaluator.fit(istl.generators.CuboidsGenerator.merge(train_split[0], train_split[1]))

		q['results'] = {}
		q['results']['2nd iteration'] = {}

		data = {0: train_split[2], 1: train_split[3]}
		val_data = {}
		for c in data:
			data[c].batch_size = p['batch_size'] if 'batch_size' in p else 1

			# The generators must return the cuboids batch as label also
			# when indexing
			data[c].return_cub_as_label = True

			# Set validation data
			data[c], val_data[c] = data[c].make_partitions((0.9, 0.1))

			# Set data augmentation
			data[c].augment_data(max_stride=3)
			data[c].shuffle(shuf=bool(p['shuffle']) if 'shuffle' in p else False,
									seed=p['seed'] if 'seed' in p else time.time())

		for c in data:
			evaluator.clear()

			meas = evaluator.evaluate_cuboids(data[c], [0]*len(data[c]))
			print('Evaluation of second iteration client set {} founding {} '\
				' false positive cuboids\n{}'.format(c, len(evaluator), meas))

			q['results']['2nd iteration'][c] = {'fp cuboids': len(evaluator),
													'measures': meas}

			if q['force_relearning']:
				data[c] = evaluator.fp_cuboids if len(evaluator) else None

		# Save the results
		results.append(q)

		with open(results_filename, 'w') as f:
			json.dump(results, f, indent=4)

		# Train with false positive cuboids
		if all(data[c] is not None for c in data):
			t_2it_start = time.time()
			print('Training on 2nd iteration - start time: {} s'.format(t_2it_start - t_start))
			print('- Client 0: {} samples, Client 1: {} samples'.format(
												len(data[0]),
												len(data[1])))

			hist2 = istl_fed_model_copy.fit(x=data, y=data, epochs=q['epochs'],
					validation_data=val_data,
					batch_size=q['batch_size'] if 'batch_size' in q else 1,
					early_stop_monitor='val_loss',
					early_stop_patience=q['early_stop_patience'] if 'early_stop_patience' in q else 5,
					early_stop_delta=q['early_stop_delta'] if 'early_stop_delta' in q else 1e-6,
					backup_filename='backup.h5',
					backup_epochs=10,
					backup_save_only_weights=False,
					verbose=2,
					shuffle=False)

			hist2_val = {c: hist2[c]['val_loss'] for c in hist2}
			hist2 = {c: hist2[c]['loss'] for c in hist2}

			t_2it_end = time.time()
			q['time']['2nd iteration'] = (t_2it_end - t_2it_start)
			print('End of training - elapsed time {} s'.format(q['time']
															['2nd iteration']))

			# Plot MSE of client training
			for c in range(2):
				plot_results({'MSE - training': hist2[c],
								'MSE - validation': hist2_val[c]},
					'Mean Squared Error - 2st iteration - client #{}'.format(c),
					model_base_filename +
					'ISTL_2st_iteration_client{}_MSE_train_loss_exp={}.pdf'.format(c, len(results)+1))

				np.savetxt(model_base_filename +
					'ISTL_2st_iteration_client{}_MSE_loss_exp={}.txt'.format(c, len(results)+1),
							hist2[c])

				np.savetxt(model_base_filename +
					'ISTL_2st_iteration_client{}_MSE_val_loss_exp={}.txt'.format(c, len(results)+1),
							hist2_val[c])

			evaluator.fit(np.concatenate(data[0], data[1]))

		else:
			hist2 = {c: [] for c in range(2)}
			hist2_val = hist2
			q['time']['2nd iteration'] = 0


		########## Third iteration (training with the 20% of data) ##########
		data = {0: train_split[4], 1: train_split[5]}
		val_data = {}
		for c in data:
			data[c].batch_size = p['batch_size'] if 'batch_size' in p else 1

			# The generators must return the cuboids batch as label also
			# when indexing
			data[c].return_cub_as_label = True

			# Set validation data
			data[c], val_data[c] = data[c].make_partitions((0.9, 0.1))

			# Set data augmentation
			data[c].augment_data(max_stride=3)
			data[c].shuffle(shuf=bool(p['shuffle']) if 'shuffle' in p else False,
									seed=p['seed'] if 'seed' in p else time.time())

		# Evaluate performance and retrieve false positive cuboids
		# to train with them
		q['results']['3rd iteration'] = {}

		for c in data:
			evaluator.clear()

			meas = evaluator.evaluate_cuboids(data[c], [0]*len(data[c]))
			print('Evaluation of third iteration client set {} founding {} false positive'\
					' cuboids\n{}'.format(c, len(evaluator), meas))

			q['results']['3rd iteration'][c] = {'fp cuboids': len(evaluator),
													'measures': meas}
			if q['force_relearning']:
				data[c] = evaluator.fp_cuboids if len(evaluator) else None

		# Save the results (p is yet stored in the list)
		with open(results_filename, 'w') as f:
			json.dump(results, f, indent=4)

		# Training with false positive cuboids
		if all(data[c] is not None for c in data):
			t_3it_start = time.time()
			print('Training on 3rd iteration - time: {} s'.format(t_3it_start - t_start))
			print('- Client 0: {} samples, Client 1: {} samples'.format(
												len(data[0]),
												len(data[1])))

			hist3 = istl_fed_model_copy.fit(x=data, y=data, epochs=q['epochs'],
					validation_data=val_data,
					batch_size=q['batch_size'] if 'batch_size' in q else 1,
					early_stop_monitor='val_loss',
					early_stop_patience=q['early_stop_patience'] if 'early_stop_patience' in q else 5,
					early_stop_delta=q['early_stop_delta'] if 'early_stop_delta' in q else 1e-6,
					backup_filename='backup.h5',
					backup_epochs=10,
					backup_save_only_weights=False,
					verbose=2,
					shuffle=False)

			hist3_val = {c: hist3[c]['val_loss'] for c in hist3}
			hist3 = {c: hist3[c]['loss'] for c in hist3}

			t_3it_end = time.time()
			q['time']['3rd iteration'] = (t_3it_end - t_3it_start)
			print('End of training - elapsed time {} s'.format(
					q['time']['3rd iteration']))

			# Plot MSE of client training
			for c in range(2):
				plot_results({'MSE - training': hist3[c],
								'MSE - validation': hist3_val[c]},
					'Mean Squared Error - 3rd iteration - client #{}'.format(c),
					model_base_filename +
					'ISTL_3rd_iteration_client{}_MSE_train_loss_exp={}.pdf'.format(c, len(results)+1))

				np.savetxt(model_base_filename +
					'ISTL_3rd_iteration_client{}_MSE_loss_exp={}.txt'.format(c, len(results)+1),
							hist3[c])

				np.savetxt(model_base_filename +
					'ISTL_3rd_iteration_client{}_MSE_val_loss_exp={}.txt'.format(c, len(results)+1),
							hist3_val[c])

			evaluator.fit(np.concatenate(data[0], data[1]))

		else:
			hist3 = {c: [] for c in range(2)}
			hist3_val = hist3
			q['time']['3rd iteration'] = 0

		# Plot MSE of all iterations
		for c in range(2):
			plot_results({'MSE - training': np.concatenate(
															(hist1[c],
															hist2[c],
															hist3[c]),
															axis=0),
						'MSE - validation': np.concatenate(
															(hist1_val[c],
															hist2_val[c],
															hist3_val[c]),
															axis=0)
																},
				'Mean Squared Error - client #{}'.format(c),
				model_base_filename +
				'ISTL_all_training_client{}_MSE_train_loss_exp={}.pdf'.format(c, len(results)))

		## Save model
		if store_models:
			istl_fed_model.global_model.save(model_base_filename +
								'-experiment-'+str(len(results)) + '_model.h5')

		## Final evaluation
		evaluator.clear()

		t_eval_start = time.time()
		meas = evaluator.evaluate_cuboids(data_test, test_labels)
		t_eval_end = time.time()

		q['time']['test evaluation'] = (t_eval_end - t_eval_start)
		q['time']['mean evaluation time per test sample'] = (
									q['time']['test evaluation'] / len(data_test))

		q['results']['test set'] = {'fp cuboids': len(evaluator),
										'measures': meas}

		print('Evaluation of test set founding {} false positive'\
					' cuboids\n{} - time taken: {}s - mean evaluation time'\
								' per test sample: {}s'.format(len(evaluator), meas,
													q['time']['test evaluation'],
								q['time']['mean evaluation time per test sample']))

		#t_end = time.time()
		q['time']['total_elapsed time'] = (q['time']['test evaluation'] +
												q['time']['3rd iteration'] +
												q['time']['2nd iteration'] +
												q['time']['1st iteration'])
		print('End of experiment - Total time taken: {}s'.format(q['time']
														['total_elapsed time']))