# -*- coding: utf-8 -*-
"""

@author: Nicolás Cubero Torres
@description: Trains and evaluate an Incremental Spatio Temporal Learner (ISTL)
    model by using the UCSD Ped 1 or the UCSD Ped 2 train and test sets on a
    federated architecture simulated on a single node.

    This scripts replicates the experiment carried out on [1] for the
    UCSD Ped 1/2 dataset, embedding it into a federated learning architecture
    in which the experiment is performed:

    The UCSD Ped 1/2 train dataset is splitted in three partitions: then
    the first partition is trained offline method
    while the two remainders partitions are trained with active learning.

    This script receives experiment description from a JSON document and saves
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

      "partitions": (list of 6 float values)
              Train samples ratio to be used for each client on each partition.
            The correspondence is:

            [1st part. - cl. #0, 1st part - cl. #1,
            2nd part. - cl. #0, 2nd part - cl. #1,
            3rd part. - cl. #0, 3rd part - cl. #1]

            For example: [0.5, 0.3, 0.03, 0.07, 0.025, 0.075]
                Sets the 50% for the client #0 on 1st iteration, the 30%
                for the the client #1 on 1st iteration, 3% for the client
                #0 on 2nd iteration, 7% for the client #1 on 2nd iteration,
                2.5% for the client #0 on 3rd iteration and 7.5% for the client
                #1 on 3rd iteration.

      "max_stride": (int), default: 3
              Maximum stride applied for train set augmentation

      "anom_thresh": (float in [0,1])|(list of floats)
              Anomaly thresholds to be used on each experiment

      "temp_thresh": (float in [0,1])|(list of ints)
              Temporal thresholds to be used on each experiment

      "force_relearning": (int containing 0 or 1)
              Whether to train with all train samples on the active learning
            iterations despite no false positive is detected (1) or train only
            with false positives found (0)

      "norm_mode": ("train_samples"    or "test_samples"), default: "train_samples"
              Use train_samples to scale reconstruction error based on training
            minimum and maximum errors and test_samples to scale based on minimum
            and maximum errors of each test sample.

      Optional:

      "epochs": (int), default: 1
              Max number of epochs performed for the training on each iteration

      "seed": (int)
              Value used as seed for the generator.

      "lr": (float), default: 1e-4
              Initial learning rate used for training

      "lr_decay": (float), default: 0.9
              Factor to reduce learning rate whether no improvement on validation
            loss is produced.

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

@usage: train_Fed_Act_ISTL_UCSDPed_val.py -d <JSON document experiment>
                                            [-s] save the resulting model
"""

# Modules imported
import argparse
import json
import sys
import time
from copy import deepcopy

import numpy as np
from cv2 import resize, cvtColor, COLOR_BGR2GRAY
from tensorflow import __version__ as tf_version
from tensorflow import config
from tensorflow import random as tf_random
from tensorflow.keras.optimizers import Adam

from fedLearn import SynFedAvgLearnModel
from learningRateImprover import LearningRateImprover
from models import istl
from utils import extract_experiments_parameters, plot_results, root_sum_squared_error

# Constants
CUBOIDS_LENGTH = 8
CUBOIDS_WIDTH = 224
CUBOIDS_HEIGHT = 224

# Image resize function
resize_fn = lambda img: np.expand_dims(resize(cvtColor(img, COLOR_BGR2GRAY), (CUBOIDS_WIDTH, CUBOIDS_HEIGHT)) / 255,
                                       axis=2)

### Input Arguments
parser = argparse.ArgumentParser(description='Trains an Incremental Spatio Temporal Learner model for the UCSD Ped 1/2 '
                                             'dataset by using active learning on a federated architecture')
parser.add_argument('-d', '--document', help='JSON file containing the train parameters', type=str)
parser.add_argument('-s', '--save_model', help='Save the resulting model on a h5 file',
                    action='store_true', default=False)

args = parser.parse_args()

exp_filename = args.document
store_models = args.save_model

# Read experiment document
with open(exp_filename) as f:
    try:
        exp_data = json.load(f)
    except Exception as e:
        print('Cannot load experiment JSON file :\n', str(e), file=sys.stderr)
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

# Data loading and preparation ###
train_video_dir = exp_data['train_video_dir']
test_video_dir = exp_data['test_video_dir']
test_label = exp_data['test_label']

data_train = istl.generators.CuboidsGeneratorFromImgs(
    source=train_video_dir,
    cub_frames=CUBOIDS_LENGTH,
    prep_fn=resize_fn,
    max_cuboids=100000)

data_test = istl.generators.CuboidsGeneratorFromImgs(source=test_video_dir,
                                                     cub_frames=CUBOIDS_LENGTH,
                                                     prep_fn=resize_fn,
                                                     max_cuboids=100000)
data_test = istl.generators.ConsecutiveCuboidsGen(data_test)
test_labels = np.loadtxt(test_label, dtype='int8')

# Configure GPU usage
physical_devices = config.experimental.list_physical_devices('GPU')
config.experimental.set_memory_growth(physical_devices[0], True)

# Perform training for each parameters combination
results = []
params = extract_experiments_parameters(exp_data, ('seed', 'batch_size'))

for p in params:

    if 'seed' in p:
        np.random.seed(p['seed'])

        if tf_version.startswith('1'):
            tf_random.set_random_seed(p['seed'])
        else:
            tf_random.set_seed(p['seed'])

    # Prepare the data train and make partitions
    train_split = data_train.make_partitions(p['partitions'])

    t_start = time.time()

    print('Training with parameters: {}'.format(p))

    #################    Model preparation    ################
    # Stochastic gradient descent algorithm
    # sgd = SGD(learning_rate=p['lr'] if 'lr' in p else 1e-2)
    adam = Adam(lr=p['lr'] if 'lr' in p else 1e-4,
                decay=p['lr_decay'] if 'lr_decay' in p else 0,
                epsilon=1e-6)

    istl_fed_model = SynFedAvgLearnModel(build_fn=istl.build_ISTL, n_clients=2,
                                         cub_length=CUBOIDS_LENGTH)
    istl_fed_model.compile(optimizer=adam, loss='mean_squared_error',
                           metrics=[root_sum_squared_error])

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
        val_data[c], data[c] = data[c].take_subpartition(
            p['port_val'] if 'port_val' in p else 0.1,
            p['seed'] if 'seed' in p else None)

        # Set data augmentation
        data[c].augment_data(max_stride=3)
        data[c].shuffle(shuf=bool(p['shuffle']) if 'shuffle' in p else False,
                        seed=p['seed'] if 'seed' in p else time.time())

    print('- Client 0: {} samples, Client 1: {} samples'.format(
        len(data[0]),
        len(data[1])))

    patience = p['patience'] if 'patience' in p else 0
    epochs = p['epochs'] if 'epochs' in p else 1
    callbacks = {c: [LearningRateImprover(
        parameter='val_loss',
        min_lr=1e-7, factor=0.9,
        patience=patience,
        min_delta=1e-6, verbose=1,
        restore_best_weights=True,
        acumulate_epochs=True)] for c in range(2)}

    hist1 = istl_fed_model.fit(x=data, epochs=epochs,
                               validation_data=val_data,
                               callbacks=callbacks,
                               backup_filename='backup.h5',
                               backup_epochs=10,
                               backup_save_only_weights=False,
                               verbose=2,
                               shuffle=False)

    hist1_rec = {c: hist1[c]['root_sum_squared_error'] if c in hist1 else [] for c in range(2)}
    hist1_val_rec = {c: hist1[c]['val_root_sum_squared_error'] if c in hist1 else [] for c in range(2)}
    hist1_val = {c: hist1[c]['val_loss'] if c in hist1 else [] for c in range(2)}
    hist1 = {c: hist1[c]['loss'] if c in hist1 else [] for c in range(2)}
    lr_hist1 = {c: callbacks[c][0].lr_history for c in callbacks}

    t_1it_end = time.time()
    p['time'] = {'1st iteration': (t_1it_end - t_1it_start)}
    print('End of 1st iteration - elapsed time {} s'.format(p['time']
                                                            ['1st iteration']))

    ## Save model of first iteration
    if store_models:
        istl_fed_model.global_model.save(model_base_filename +
                                         '-experiment-' + str(len(results) + 1) + '_1st_iteration_model.h5')

    # Plot MSE of client training
    for c in range(2):
        plot_results({'MSE - training': hist1[c],
                      'MSE - validation': hist1_val[c]},
                     f'Mean Squared Error - 1st iteration - client #{c}',
                     model_base_filename +
                     f'ISTL_1st_iteration_client{c}_MSE_train_loss_exp={len(results) + 1}.pdf')

        plot_results({'RSSE - training': hist1_rec[c],
                      'RSSE - validation': hist1_val_rec[c]},
                     f'Root of the Sum of Squared Errors - 1st iteration - client #{c}',
                     model_base_filename +
                     f'ISTL_1st_iteration_client{c}_RSSE_train_loss_exp={len(results) + 1}.pdf')

        np.savetxt(model_base_filename +
                   f'ISTL_1st_iteration_client{c}_MSE_loss_exp={len(results) + 1}.txt',
                   hist1[c])

        np.savetxt(model_base_filename +
                   f'ISTL_1st_iteration_client{c}_MSE_val_loss_exp={len(results) + 1}.txt',
                   hist1_val[c])

        np.savetxt(model_base_filename +
                   f'ISTL_1st_iteration_client{c}_RSSE_loss_exp={len(results) + 1}.txt',
                   hist1_rec[c])

        np.savetxt(model_base_filename +
                   f'ISTL_1st_iteration_client{c}_RSSE_val_loss_exp={len(results) + 1}.txt',
                   hist1_val_rec[c])

        # Plot lr history
        plot_results({'Lr history': lr_hist1[c]},
                     'Learning rate history',
                     model_base_filename +
                     f'ISTL_1st_iteration_lr_history_client={c}_exp={len(results) + 1}.pdf')

        np.savetxt(model_base_filename +
                   f'ISTL_1st_iteration_lr_history_client={c}_exp={len(results) + 1}.txt',
                   lr_hist1[c])

    ## Perform the second and third iteration for each combination of anomaly
    ## thresholds and temporal thresholds
    thresh_params = extract_experiments_parameters(p, ('anom_thresh',
                                                       'temp_thresh',
                                                       'force_relearning',
                                                       'norm_mode'))
    for q in thresh_params:

        istl_fed_model_copy = deepcopy(istl_fed_model)

        ### Evaluation of test set after 1st iteration
        # Prepare to save results
        q['results'] = {}
        q['results']['test set'] = {}

        evaluator = istl.EvaluatorISTL(model=istl_fed_model_copy.global_model,
                                       cub_frames=CUBOIDS_LENGTH,
                                       anom_thresh=q['anom_thresh'],
                                       temp_thresh=q['temp_thresh'])

        for split in (train_split[0], train_split[1]):
            split.batch_size = 1
            split.return_cub_as_label = False

        train_eval = istl.generators.CuboidsGenerator.merge(train_split[0],
                                                            train_split[1])

        if q['norm_mode'] == 'train_samples':
            train_rec_error = evaluator.fit(train_eval)
        elif q['norm_mode'] == 'test_samples':
            train_rec_error = evaluator.score_cuboids(train_eval, False)
        else:
            print('Unknow "{}" norm mode'.format(q['norm_mode']),
                  file=sys.stderr)
            exit(-1)

        q['training_rec_errors'] = {'1st iteration': {
            'mean': float(train_rec_error.mean()),
            'std': float(train_rec_error.std()),
            'min': float(train_rec_error.min()),
            'max': float(train_rec_error.max())
        }}

        # Fit the evaluator to the train samples if this normalization
        #    mode is set
        cum_cuboids_per_video = data_test.cum_cuboids_per_video if q['norm_mode'] == 'test_samples' else None
        meas = evaluator.evaluate_cuboids(data_test, test_labels,
                                          cum_cuboids_per_video)

        q['results']['test set']['1st iteration'] = {
            'fp cuboids': len(evaluator),
            'measures': meas
        }

        print('Executing 2nd and 3rd iteration with parameters: {}'.format(q))

        q['#experiment'] = len(results) + 1
        q['force_relearning'] = (bool(q['force_relearning'])
                                 if 'force_relearning' in q else False)

        ########## Second iteration (training with the 20% of data) ###########

        # Evaluate performance and retrieve false positive cuboids
        # to train with them

        # q['results']['2nd iteration'] = {}
        q['active_training_evaluation'] = {}
        q['active_training_evaluation']['2nd iteration'] = {}

        data = {0: train_split[2], 1: train_split[3]}
        val_data = {}
        for c in data:
            # data[c].batch_size = p['batch_size'] if 'batch_size' in p else 1

            # The generators must return the cuboids batch as label also
            # when indexing
            data[c].return_cub_as_label = False

            # Set validation data
            val_data[c], data[c] = data[c].take_subpartition(
                p['port_val'] if 'port_val' in p else 0.1,
                p['seed'] if 'seed' in p else None)

            if len(val_data[c]) > 1:
                val_data[c] = np.array([val_data[c][i][0] for i in range(len(val_data[c]))])
            else:
                val_data[c] = val_data[c][0]

            val_data[c] = (val_data[c],) * 2

            # Set data augmentation
            data[c].augment_data(max_stride=3)

        for c in data:
            evaluator.clear()

            train_test = istl.generators.ConsecutiveCuboidsGen(data[c])
            cum_cuboids_per_video = train_test.cum_cuboids_per_video if q['norm_mode'] == 'test_samples' else None
            meas = evaluator.evaluate_cuboids(train_test, [0] * len(train_test),
                                              cum_cuboids_per_video)
            print(f'Evaluation of second iteration client set {c} founding {len(evaluator)} '
                  f'false positive cuboids\n{meas}')

            q['active_training_evaluation']['2nd iteration'][c] = {
                'fp cuboids': len(evaluator),
                'measures': meas}

            if not q['force_relearning']:
                data[c] = evaluator.fp_cuboids  # if len(evaluator) else None
            else:
                print('Training with all samples despite no false positive has been found')
                if len(data[c]) > 1:
                    data[c] = np.array([data[c][i][0] for i in range(len(data[c]))])
                else:
                    data[c] = data[c][0]

        # Save the results
        results.append(q)

        with open(results_filename.format(len(results)), 'w') as f:
            json.dump(q, f, indent=4)

        # Train with false positive cuboids
        if any(data[c] is not None for c in data):
            t_2it_start = time.time()
            print(f'Training on 2nd iteration - start time: {t_2it_start - t_start} s')
            print(f'- Client 0: {len(data[0]) if data[0] is not None else 0} samples, '
                  f'Client 1: {len(data[1]) if data[1] is not None else 0} samples')

            patience = p['patience'] if 'patience' in p else 0
            epochs = p['epochs'] if 'epochs' in p else 1
            callbacks = {c: [LearningRateImprover(
                initial_value=p['lr'] if 'lr' in p else 1e-4,
                parameter='val_loss',
                min_lr=1e-7, factor=0.9,
                patience=patience,
                min_delta=1e-6, verbose=1,
                restore_best_weights=True,
                acumulate_epochs=True)] for c in range(2)}

            hist2 = istl_fed_model_copy.fit(x=data, y=data, epochs=epochs,
                                            validation_data=val_data,
                                            batch_size=q['batch_size'] if 'batch_size' in q else 1,
                                            callbacks=callbacks,
                                            backup_filename='backup.h5',
                                            backup_epochs=10,
                                            backup_save_only_weights=False,
                                            verbose=2,
                                            shuffle=False)

            hist2_rec = {c: hist2[c]['root_sum_squared_error'] if c in hist2 else [] for c in range(2)}
            hist2_val_rec = {c: hist2[c]['val_root_sum_squared_error'] if c in hist2 else [] for c in range(2)}
            hist2_val = {c: hist2[c]['val_loss'] if c in hist2 else [] for c in range(2)}
            hist2 = {c: hist2[c]['loss'] if c in hist2 else [] for c in range(2)}
            lr_hist2 = {c: callbacks[c][0].lr_history for c in callbacks}

            t_2it_end = time.time()
            q['time']['2nd iteration'] = (t_2it_end - t_2it_start)
            print(f'End of training - elapsed time {q["time"]["2nd iteration"]} s')

            # Save model
            if store_models:
                istl_fed_model_copy.global_model.save(model_base_filename +
                                                      '-experiment-' + str(len(results)) + '_2nd_iteration_model.h5')

            # Plot MSE of client training
            for c in range(2):
                plot_results({'MSE - training': hist2[c],
                              'MSE - validation': hist2_val[c]},
                             'Mean Squared Error - 2nd iteration - client #{}'.format(c),
                             model_base_filename +
                             f'ISTL_2nd_iteration_client{c}_MSE_train_loss_exp={len(results) + 1}.pdf')

                plot_results({'RSSE - training': hist2_rec[c],
                              'RSSE - validation': hist2_val_rec[c]},
                             'Root of the Sum of Squared Errors - 2nd iteration - client #{}'.format(c),
                             model_base_filename +
                             f'ISTL_2nd_iteration_client{c}_RSSE_train_loss_exp={len(results) + 1}.pdf')

                np.savetxt(model_base_filename +
                           f'ISTL_2st_iteration_client{c}_MSE_loss_exp={len(results) + 1}.txt',
                           hist2[c])

                np.savetxt(model_base_filename +
                           f'ISTL_2st_iteration_client{c}_MSE_val_loss_exp={len(results) + 1}.txt',
                           hist2_val[c])

                np.savetxt(model_base_filename +
                           f'ISTL_2nd_iteration_client{c}_RSSE_loss_exp={len(results) + 1}.txt',
                           hist2_rec[c])

                np.savetxt(model_base_filename +
                           f'ISTL_2nd_iteration_client{c}_RSSE_val_loss_exp={len(results) + 1}.txt',
                           hist2_val_rec[c])

                # Plot lr history
                plot_results({'Lr history': lr_hist2[c]},
                             'Learning rate history',
                             model_base_filename +
                             f'ISTL_2nd_iteration_lr_history_client={c}_exp={len(results) + 1}.pdf')

                np.savetxt(model_base_filename +
                           f'ISTL_2nd_iteration_lr_history_client={c}_exp={len(results) + 1}.txt',
                           lr_hist2[c])

            train_eval = istl.generators.CuboidsGenerator.merge(train_split[0],
                                                                train_split[1],
                                                                train_split[2],
                                                                train_split[3])

            if q['norm_mode'] == 'train_samples':
                train_rec_error = evaluator.fit(train_eval)
            elif q['norm_mode'] == 'test_samples':
                train_rec_error = evaluator.score_cuboids(train_eval, False)
            else:
                print(f'Unknow "{q["norm_mode"]}" norm mode',
                      file=sys.stderr)

            q['training_rec_errors']['2nd iteration'] = {
                'mean': float(train_rec_error.mean()),
                'std': float(train_rec_error.std()),
                'min': float(train_rec_error.min()),
                'max': float(train_rec_error.max())
            }

            ### Evaluation of test set
            evaluator.clear()
            cum_cuboids_per_video = data_test.cum_cuboids_per_video if q['norm_mode'] == 'test_samples' else None
            meas = evaluator.evaluate_cuboids(data_test, test_labels,
                                              cum_cuboids_per_video)

            q['results']['test set']['2nd iteration'] = {
                'fp cuboids': len(evaluator),
                'measures': meas
            }

        else:
            hist2 = {c: [] for c in range(2)}
            hist2_val = hist2
            hist2_rec = hist2
            hist2_val_rec = hist2
            lr_hist2 = hist2
            q['time']['2nd iteration'] = 0

        ########## Third iteration (training with the 20% of data) ##########
        data = {0: train_split[4], 1: train_split[5]}
        val_data = {}
        for c in data:
            # data[c].batch_size = p['batch_size'] if 'batch_size' in p else 1

            # The generators must return the cuboids batch as label also
            # when indexing
            data[c].return_cub_as_label = False

            # Set validation data
            val_data[c], data[c] = data[c].take_subpartition(
                p['port_val'] if 'port_val' in p else 0.1,
                p['seed'] if 'seed' in p else None)

            if len(val_data[c]) > 1:
                val_data[c] = np.array([val_data[c][i][0] for i in range(len(val_data[c]))])
            else:
                val_data[c] = val_data[c][0]

            val_data[c] = (val_data[c],) * 2

            # Set data augmentation
            data[c].augment_data(max_stride=3)

        # Evaluate performance and retrieve false positive cuboids
        # to train with them
        q['active_training_evaluation']['3rd iteration'] = {}

        for c in data:
            evaluator.clear()

            train_test = istl.generators.ConsecutiveCuboidsGen(data[c])
            cum_cuboids_per_video = train_test.cum_cuboids_per_video if q['norm_mode'] == 'test_samples' else None
            meas = evaluator.evaluate_cuboids(train_test, [0] * len(train_test),
                                              cum_cuboids_per_video)
            print(f'Evaluation of third iteration client set {c} founding {len(evaluator)} false positive '
                  f'cuboids\n{meas}')

            q['active_training_evaluation']['3rd iteration'][c] = {
                'fp cuboids': len(evaluator),
                'measures': meas
            }
            if not q['force_relearning']:
                data[c] = evaluator.fp_cuboids  # if len(evaluator) else None
            else:
                print('Training with all samples despite no false positive has been found')
                if len(data[c]) > 1:
                    data[c] = np.array([data[c][i][0] for i in range(len(data[c]))])
                else:
                    data[c] = data[c][0]

        # Save the results (p is yet stored in the list)
        with open(results_filename.format(len(results)), 'w') as f:
            json.dump(q, f, indent=4)

        # Training with false positive cuboids
        if any(data[c] is not None for c in data):
            t_3it_start = time.time()
            print(f'Training on 3rd iteration - time: {t_3it_start - t_start} s')
            print(f'- Client 0: {len(data[0]) if data[0] is not None else 0} samples, '
                  f'Client 1: {len(data[1]) if data[1] is not None else 0} samples')

            patience = p['patience'] if 'patience' in p else 0
            epochs = p['epochs'] if 'epochs' in p else 1
            callbacks = {c: [LearningRateImprover(
                initial_value=p['lr'] if 'lr' in p else 1e-4,
                parameter='val_loss',
                min_lr=1e-7, factor=0.9,
                patience=patience,
                min_delta=1e-6, verbose=1,
                restore_best_weights=True,
                acumulate_epochs=True)] for c in range(2)}

            hist3 = istl_fed_model_copy.fit(x=data, y=data, epochs=epochs,
                                            validation_data=val_data,
                                            batch_size=q['batch_size'] if 'batch_size' in q else 1,
                                            callbacks=callbacks,
                                            backup_filename='backup.h5',
                                            backup_epochs=10,
                                            backup_save_only_weights=False,
                                            verbose=2,
                                            shuffle=False)

            hist3_rec = {c: hist3[c]['root_sum_squared_error'] if c in hist3 else [] for c in range(2)}
            hist3_val_rec = {c: hist3[c]['val_root_sum_squared_error'] if c in hist3 else [] for c in range(2)}
            hist3_val = {c: hist3[c]['val_loss'] if c in hist3 else [] for c in range(2)}
            hist3 = {c: hist3[c]['loss'] if c in hist3 else [] for c in range(2)}
            lr_hist3 = {c: callbacks[c][0].lr_history for c in callbacks}

            t_3it_end = time.time()
            q['time']['3rd iteration'] = (t_3it_end - t_3it_start)
            print('End of training - elapsed time {} s'.format(
                q['time']['3rd iteration']))

            # Plot MSE of client training
            for c in range(2):
                plot_results({'MSE - training': hist3[c],
                              'MSE - validation': hist3_val[c]},
                             f'Mean Squared Error - 3rd iteration - client #{c}',
                             model_base_filename +
                             f'ISTL_3rd_iteration_client{c}_MSE_train_loss_exp={len(results) + 1}.pdf')

                plot_results({'RSSE - training': hist3_rec[c],
                              'RSSE - validation': hist3_val_rec[c]},
                             f'Root of the Sum of Squared Errors - 3rd iteration - client #{c}',
                             model_base_filename +
                             f'ISTL_3rd_iteration_client{c}_RSSE_train_loss_exp={len(results) + 1}.pdf')

                np.savetxt(model_base_filename +
                           f'ISTL_3rd_iteration_client{c}_MSE_loss_exp={len(results) + 1}.txt',
                           hist3[c])

                np.savetxt(model_base_filename +
                           f'ISTL_3rd_iteration_client{c}_MSE_val_loss_exp={len(results) + 1}.txt',
                           hist3_val[c])

                np.savetxt(model_base_filename +
                           f'ISTL_3rd_iteration_client{c}_RSSE_loss_exp={len(results) + 1}.txt',
                           hist3_rec[c])

                np.savetxt(model_base_filename +
                           f'ISTL_3rd_iteration_client{c}_RSSE_val_loss_exp={len(results) + 1}.txt',
                           hist3_val_rec[c])

                # Plot lr history
                plot_results({'Lr history': lr_hist3[c]},
                             'Learning rate history',
                             model_base_filename +
                             f'ISTL_3rd_iteration_lr_history_client={c}_exp={len(results) + 1}.pdf')

                np.savetxt(model_base_filename +
                           f'ISTL_3rd_iteration_lr_history_client={c}_exp={len(results) + 1}.txt',
                           lr_hist3[c])

            if q['norm_mode'] == 'train_samples':
                train_rec_error = evaluator.fit(data_train)
            elif q['norm_mode'] == 'test_samples':
                train_rec_error = evaluator.score_cuboids(data_train, False)
            else:
                print('Unknow "{}" norm mode'.format(q['norm_mode']),
                      file=sys.stderr)

            q['training_rec_errors']['3rd iteration'] = {
                'mean': float(train_rec_error.mean()),
                'std': float(train_rec_error.std()),
                'min': float(train_rec_error.min()),
                'max': float(train_rec_error.max())
            }

        else:
            hist3 = {c: [] for c in range(2)}
            hist3_val = hist3
            hist3_rec = hist3
            hist3_val_rec = hist3
            lr_hist3 = hist3
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
                f'Mean Squared Error - client #{c}',
                model_base_filename +
                f'ISTL_all_training_client{c}_MSE_train_loss_exp={len(results)}.pdf')

            plot_results({'RSSE - training': np.concatenate(
                (hist1_rec[c],
                 hist2_rec[c],
                 hist3_rec[c]),
                axis=0),
                'RSSE - validation': np.concatenate(
                    (hist1_val_rec[c],
                     hist2_val_rec[c],
                     hist3_val_rec[c]),
                    axis=0)
            },
                f'Root of the Sum of Squared Errors - client #{c}',
                model_base_filename +
                f'ISTL_all_training_client{c}_RSSE_train_loss_exp={len(results)}.pdf')

        ## Save definitive model
        if store_models:
            istl_fed_model_copy.global_model.save(model_base_filename +
                                                  '-experiment-' + str(len(results)) + '_model.h5')

        ## Final evaluation
        evaluator.clear()

        cum_cuboids_per_video = data_test.cum_cuboids_per_video if q['norm_mode'] == 'test_samples' else None

        t_eval_start = time.time()
        meas = evaluator.evaluate_cuboids(data_test, test_labels,
                                          cum_cuboids_per_video)
        t_eval_end = time.time()

        q['time']['test evaluation'] = (t_eval_end - t_eval_start)
        q['time']['mean evaluation time per test sample'] = (
                q['time']['test evaluation'] / len(data_test))

        q['results']['test set']['3rd iteration'] = {
            'fp cuboids': len(evaluator),
            'measures': meas
        }

        print(f'Evaluation of test set founding {len(evaluator)} false positive cuboids\n{meas} - '
              f'time taken: {q["time"]["test evaluation"]}s - '
              f'mean evaluation time per test sample: {q["time"]["mean evaluation time per test sample"]}s')

        # t_end = time.time()
        q['time']['total_elapsed time'] = (q['time']['test evaluation'] +
                                           q['time']['3rd iteration'] +
                                           q['time']['2nd iteration'] +
                                           q['time']['1st iteration'])
        print(f'End of experiment - Total time taken: {q["time"]["total_elapsed time"]}s')

        # Save the results (p is yet stored in the list)
        with open(results_filename.format(len(results)), 'w') as f:
            json.dump(q, f, indent=4)
