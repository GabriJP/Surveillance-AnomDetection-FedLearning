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
import json
import sys
from pathlib import Path

import click
import numpy as np
from cv2 import resize, cvtColor, COLOR_BGR2GRAY
from tensorflow import config
from tensorflow.keras.models import load_model

from models import istl
from utils import root_sum_squared_error

physical_devices = config.experimental.list_physical_devices('GPU')
if len(physical_devices):
    config.experimental.set_memory_growth(physical_devices[0], True)

# Constants
CUBOIDS_LENGTH = 8
CUBOIDS_WIDTH = 224
CUBOIDS_HEIGHT = 224


# Image resize function
def resize_fn(img):
    np.expand_dims(resize(cvtColor(img, COLOR_BGR2GRAY), (CUBOIDS_WIDTH, CUBOIDS_HEIGHT)) / 255, axis=2)


@click.command()
@click.option('-m', '--model', help='A pretrained model stored on a h5 file', type=str)
@click.option('-c', '--train_folder', help='Path to folder containing the train dataset',
              type=click.Path(exists=True, file_okay=False, resolve_path=True))
@click.option('-d', '--data_folder', help='Path to folder containing the test dataset',
              type=click.Path(exists=True, file_okay=False, resolve_path=True))
@click.option('-l', '--labels', help='Path to file containing the test labels',
              type=click.Path(exists=True, dir_okay=False, resolve_path=True))
@click.option('-a', '--anom_threshold', help='Anomaly threshold values to test', type=float)
@click.option('-t', '--temp_threshold', help='Temporal threshold values to test', type=int)
@click.option('-o', '--output', help='Output file in which the results will be located',
              type=click.Path(dir_okay=False, resolve_path=True))
@click.option('-n', '--norm_zero_one',
              help='Set the normalization between zero or one. Only valid for the evaluation per test sample',
              is_flag=True)
@click.option('--train_cons_cuboids', help='Set the usage of overlapping cuboids for the training set',
              is_flag=True)
def main(model, train_folder, data_folder, labels, anom_threshold, temp_threshold, output, norm_zero_one,
         train_cons_cuboids):
    """Test an Incremental Spatio Temporal Learner model for a given test dataset"""
    output = Path(output)
    results_fn = output.parent / f'{output.name}.json'

    # Loads model
    try:
        model = load_model(model, custom_objects=dict(root_sum_squared_error=root_sum_squared_error))
    except Exception as e:
        print(f'Cannot load the model: {e}', file=sys.stderr)
        exit(-1)

    # Load the video test dataset
    data_train = None
    if train_folder:
        try:
            data_train = istl.generators.CuboidsGeneratorFromImgs(source=train_folder, cub_frames=CUBOIDS_LENGTH,
                                                                  prep_fn=resize_fn)

            # Use overlapping cuboids for training set
            if train_cons_cuboids:
                data_train = istl.generators.ConsecutiveCuboidsGen(data_train)

        except Exception as e:
            print(f'Cannot load {train_folder}: ', str(e), file=sys.stderr)
            exit(-1)

    try:
        data_test = istl.generators.CuboidsGeneratorFromImgs(source=data_folder, cub_frames=CUBOIDS_LENGTH,
                                                             prep_fn=resize_fn)
        data_test = istl.generators.ConsecutiveCuboidsGen(data_test)
    except Exception as e:
        print(f'Cannot load {data_folder}: {e}', file=sys.stderr)
        return

    try:
        test_labels = np.loadtxt(labels, dtype='int8')
    except Exception as e:
        print(f'Cannot load {labels}: {e}', file=sys.stderr)
        return

        # Testing for each pair of anomaly and temporal values combinations
    print('Performing evaluation with all anomaly and temporal thesholds combinations')
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
        return

    if data_train is not None:
        all_meas['training_rec_error'] = dict(
            mean=float(sc_train.mean()),
            std=float(sc_train.std()),
            min=float(sc_train.min()),
            max=float(sc_train.max()),
        )

    # Save the results
    with results_fn.open(mode='w') as f:
        json.dump(all_meas, f, indent=4)


if __name__ == '__main__':
    main()
