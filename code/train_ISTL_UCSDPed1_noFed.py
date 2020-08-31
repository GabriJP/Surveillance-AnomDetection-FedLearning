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
import json
#from copy import deepcopy
import numpy as np
from tensorflow.keras.models import clone_model, load_model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import MeanSquaredError
from tensorflow import config
from cv2 import resize, cvtColor, COLOR_BGR2GRAY
from utils import extract_experiments_parameters, plot_results
#from fedLearn import SynFedAvgLearnModel
from models import istl

#from tensorflow.keras import backend as K

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
	data_train.shuffle(shuf=bool(p['shuffle']) if 'shuffle' in p else False,
							seed=p['seed'] if 'seed' in p else time.time())

	train_split = data_train.make_partitions((0.6, 0.2, 0.2))

	t_start = time.time()

	print('Training with parameters: {}'.format(p))

	#################    Model preparation    ################
	# Stochastic gradient descent algorithm
	#sgd = SGD(learning_rate=p['lr'] if 'lr' in p else 1e-2)
	#adam = Adam(lr=1e-4, decay=1e-5, epsilon=1e-6)

	"""
	istl_model = istl.build_ISTL(cub_length=CUBOIDS_LENGTH)
	istl_model.compile(optimizer=sgd, loss=MeanSquaredError())
	"""
	istl_model = load_model('/home/ncubero/experimentos/experimentISTL_UCSDPed1_noFed-23/experimentISTL_UCSDPed1_noFed-23-experiment-1_model.h5')




	########## First iteration (training with the 60% of data) ##########
	"""
	t_1it_start = time.time()
	print('Training on 1st iteration')

	data = train_split[0]
	data.augment_data(max_stride=2)

	print('- {} samples'.format(len(data)))

	# The generators must return the cuboids batch as label also when indexing
	data.return_cub_as_label = True
	data.batch_size = p['batch_size'] if 'batch_size' in p else 1

	hist1 = istl_model.fit(x=data, epochs=p['epochs'],
						callbacks=[EarlyStopping(monitor='loss', patience=5,
												min_delta=1e-6)],
						verbose=2,
						shuffle=False)
	hist1 = hist1.history['loss']

	t_1it_end = time.time()
	p['time'] = {'1st iteration': (t_1it_end - t_1it_start)}
	print('End of 1st iteration - elapsed time {} s'.format(p['time']
															['1st iteration']))

	# Plot MSE
	plot_results({'MSE - 1st iteration': hist1},
		'Mean Squared Error',
		model_base_filename +
		'ISTL_UCSDPed1_1st_iteration_MSE_train_loss_exp={}.pdf'.format(len(results)+1))

	np.savetxt(model_base_filename +
		'ISTL_UCSDPed1_1st_iteration_MSE_train_loss_exp={}.txt'.format(len(results)+1),
				hist1)
	"""
	p['time'] = {}
	## Perform the second and third iteration for each combination of anomaly
	## thresholds and temporal thresholds
	thresh_params = extract_experiments_parameters(p, ('anom_thresh',
														'temp_thresh'))

	for q in thresh_params:

		#sgd = SGD(learning_rate=p['lr'] if 'lr' in p else 1e-2)
		adam = Adam(lr=1e-4, decay=1e-5, epsilon=1e-6)
		istl_model_copy = clone_model(istl_model)
		istl_model_copy.compile(optimizer=adam, loss=MeanSquaredError())

		q['#experiment'] = len(results) + 1

		########## Second iteration (training with the 20% of data) ###########

		evaluator = istl.EvaluatorISTL(model=istl_model_copy,
										cub_frames=CUBOIDS_LENGTH,
										anom_thresh=q['anom_thresh'],
										temp_thresh=q['temp_thresh'])

		# Fit the evaluator to the scores evaluated for the training set
		#train_split[0].return_cub_as_label = False
		#train_split[0].batch_size = 1
		#evaluator.fit(istl.generators.ConsecutiveCuboidsGen(train_split[1]))

		# Evaluate performance and retrieve false positive cuboids
		# to train with them
		q['results'] = {}
		q['results']['2nd iteration'] = {}
		data = istl.generators.ConsecutiveCuboidsGen(train_split[1])

		evaluator.clear()

		meas = evaluator.evaluate_cuboids(data, [0]*len(data))
		print('Evaluation of second iteration founding {} '\
			' false positive cuboids\n{}'.format(len(evaluator), meas))

		q['results']['2nd iteration'] = {'fp cuboids': len(evaluator),
												'measures': meas}

		data = evaluator.fp_cuboids if len(evaluator) else None

		# Save the results
		results.append(q)

		with open(results_filename, 'w') as f:
			json.dump(results, f, indent=4)

		if data is not None:
			# Train with false positive cuboids
			t_2it_start = time.time()
			print('Training on 2nd iteration - start time: {} s'.format(
														t_2it_start - t_start))
			print('- {} samples'.format(len(data)*(p['batch_size']
												if 'batch_size' in p else 1)))

			hist2 = istl_model_copy.fit(x=data, y=data, epochs=q['epochs'],
					batch_size=q['batch_size'] if 'batch_size' in q else None,
						callbacks=[EarlyStopping(monitor='loss', patience=5,
								min_delta=1e-6, restore_best_weights=True)],
						verbose=2,
						shuffle=False)

			hist2 = hist2.history['loss']

			t_2it_end = time.time()
			q['time']['2nd iteration'] = (t_2it_end - t_2it_start)
			print('End of training - elapsed time {} s'.format(q['time']
															['2nd iteration']))


			# Plot MSE of training
			plot_results({'MSE - 2nd iteration': hist2},
				'Mean Squared Error',
				model_base_filename+
				'ISTL_UCSDPed1_2nd_iteration_MSE_train_loss_exp={}.pdf'.format(
																len(results)))

			np.savetxt(model_base_filename +
				'ISTL_UCSDPed1_2nd_iteration_MSE_train_loss_exp={}.txt'.format(
																len(results)),
						hist2)
		else:
			hist2 = []
			q['time']['2nd iteration'] = 0
			



		########## Third iteration (training with the 20% of data) ##########
		data = istl.generators.ConsecutiveCuboidsGen(train_split[2])

		# Evaluate performance and retrieve false positive cuboids
		# to train with them
		q['results']['3rd iteration'] = {}

		evaluator.clear()
		#evaluator.fit(data)
		meas = evaluator.evaluate_cuboids(data, [0]*len(data))
		print('Evaluation of third iteration founding {} false positive'\
				' cuboids\n{}'.format(len(evaluator), meas))

		q['results']['3rd iteration'] = {'fp cuboids': len(evaluator),
												'measures': meas}

		data = evaluator.fp_cuboids if len(evaluator) else None

		# Save the results (p is yet stored in the list)
		with open(results_filename, 'w') as f:
			json.dump(results, f, indent=4)

		if data is not None:
			# Training with false positive cuboids
			t_3it_start = time.time()
			print('Training on 3rd iteration - time: {} s'.format(t_3it_start -
																		t_start))
			print('- {} samples'.format(len(data)*(p['batch_size']
												if 'batch_size' in p else 1)))

			hist3 = istl_model_copy.fit(x=data, y=data, epochs=q['epochs'],
				batch_size=q['batch_size'] if 'batch_size' in q else None,
				callbacks=[EarlyStopping(monitor='loss', patience=5,
						min_delta=1e-6, restore_best_weights=True)],
				verbose=2,
				shuffle=False)

			hist3 = hist3.history['loss']

			t_3it_end = time.time()
			q['time']['3rd iteration'] = (t_3it_end - t_3it_start)
			print('End of training - elapsed time {} s'.format(
					q['time']['3rd iteration']))


			# Plot MSE of client training
			plot_results({'MSE - 3rd iteration': hist3},
				'Mean Squared Error',
				model_base_filename +
				'ISTL_UCSDPed1_3rd_iteration_MSE_train_loss_exp={}.pdf'.format(
																len(results)))

			np.savetxt(model_base_filename +
				'ISTL_UCSDPed1_3rd_iteration_MSE_train_loss_exp={}.txt'.format(
																len(results)),
						hist3)
		else:
			hist3 = []
			q['time']['3rd iteration'] = 0

		# Plot MSE of all iterations
		plot_results({'MSE - training': np.concatenate((hist2, hist3), #hist1,
														axis=0)},
			'Mean Squared Error',
			model_base_filename +
			'ISTL_UCSDPed1_all_training_MSE_train_loss_exp={}.pdf'.format(
																len(results)))

		## Save model
		if store_models:
			istl_model.save(model_base_filename +
								'-experiment-'+str(len(results)) + '_model.h5')

		## Final evaluation
		evaluator.clear()
		#evaluator.fit(data_test)

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
								' per test sample: {}s'.format(len(evaluator),
											meas, q['time']['test evaluation'],
							q['time']['mean evaluation time per test sample']))

		#t_end = time.time()
		q['time']['total_elapsed time'] = (q['time']['test evaluation'] +
												q['time']['3rd iteration'] +
												q['time']['2nd iteration']) #+
												#q['time']['1st iteration'])
		print('End of experiment - Total time taken: {}s'.format(q['time']
														['total_elapsed time']))

		# Save the results (p is yet stored in the list)
		with open(results_filename, 'w') as f:
			json.dump(results, f, indent=4)

		## Test all combinations of anomaly and temporal threshold
		"""
		print('Performing evaluation with all anomaly and temporal '\
				'thesholds combinations')
		all_meas = evaluator.evaluate_cuboids_range_params(data_test,
												test_labels,
												np.arange(0.01, 0.61, 0.01),
												np.arange(1,10))
		q['results']['test all combinations'] = all_meas

		# Save the results (p is yet stored in the list)
		with open(results_filename, 'w') as f:
			json.dump(results, f, indent=4)
		"""
