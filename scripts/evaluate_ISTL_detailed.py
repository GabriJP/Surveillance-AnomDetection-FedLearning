# -*- coding: utf-8 -*-
"""

@author: Nicolás Cubero Torres
@description: Evaluate the prediction of an ISTL model for a given test dataset.

@usage: evaluate_ISTL_detailed.py -m <Pretrained model's h5 file>
                    -d <Directory Path containing the test set to evaluate>
                    -l <File containing the test labels>
                    -a <Anomaly threshold values to evaluate>
                    -t <Temporal threshold values to evaluate>
                    -o <Output directory>
                    [-c <Directory Path containing the train set used for
                        fitting the reconstruction error scaling>]
                    [-s] Perform spatial location over the test samples
"""
# Modules imported
import json
import sys
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
from cv2 import imwrite, resize, cvtColor, COLOR_BGR2GRAY, COLOR_GRAY2BGR, rectangle, addWeighted
from tensorflow import config
from tensorflow.keras.models import load_model

from models import istl
from utils import root_sum_squared_error, split_measures_per_video

physical_devices = config.experimental.list_physical_devices('GPU')
config.experimental.set_memory_growth(physical_devices[0], True)

# Constants
CUBOIDS_LENGTH = 8
CUBOIDS_WIDTH = 224
CUBOIDS_HEIGHT = 224

SUBWIND_WIDTH = 16
SUBWIND_HEIGHT = 16

SUBWIND_TP = np.zeros((CUBOIDS_WIDTH, CUBOIDS_HEIGHT, 3), dtype=np.uint8)
SUBWIND_TP[:, :, 0] = 255
SUBWIND_TP[:, :, 1] = 188

SUBWIND_FP = np.zeros((CUBOIDS_WIDTH, CUBOIDS_HEIGHT, 3), dtype=np.uint8)
SUBWIND_FP[:, :, 0] = 255


# Image resize function
def resize_fn(img):
    np.expand_dims(resize(cvtColor(img, COLOR_BGR2GRAY), (CUBOIDS_WIDTH, CUBOIDS_HEIGHT)) / 255, axis=2)


def group_points_in_ranges(points: np.array):
    # Check input
    if not isinstance(points, np.ndarray):
        raise TypeError('points must be a numpy array')

    if points.size == 0 or points.ndim != 1:
        raise ValueError('points must be 1D array with data')

    ranges = list()

    start_idx = 0
    idx = start_idx

    while idx < points.size:
        if points[idx] != points[start_idx]:
            # Register the limits of each group
            ranges.append((points[start_idx], (start_idx, idx - 1)))

            start_idx = idx

        idx += 1
    else:
        ranges.append((points[start_idx], (start_idx, idx - 1)))

    return ranges


@click.command()
@click.option('-m', '--model', help='A pretrained model stored on a h5 file', type=str)
@click.option('-a', '--anom_threshold',
              help='Anomaly threshold values to evaluate', type=float, nargs=1)
@click.option('-t', '--temp_threshold',
              help='Temporal threshold values to evaluate', type=int, nargs=1)
@click.option('-o', '--output', help='Output file in which the results will be located', type=str)
@click.option('-s', '--spatial_location', help='Perform spatial location of cuboids', is_flag=True)
def main(model, train_folder, data_folder, labels, anom_threshold, temp_threshold, output, spatial_location):
    """Test an Incremental Spatio Temporal Learner model for a given test dataset"""
    # Create output directory
    output = Path(output)
    output.mkdir(exist_ok=True)

    # Loads model
    try:
        model = load_model(model, custom_objects=dict(root_sum_squared_error=root_sum_squared_error))
    except Exception as e:
        print(f'Cannot load the model: {e}', file=sys.stderr)
        exit(-1)

    # Load the video test dataset
    if train_folder:
        try:
            data_train = istl.generators.CuboidsGeneratorFromImgs(
                source=train_folder,
                cub_frames=CUBOIDS_LENGTH,
                prep_fn=resize_fn)
        except Exception as e:
            print(f'Cannot load {train_folder}: {e}', file=sys.stderr)
            exit(-1)
    else:
        data_train = None

    try:
        data_test = istl.generators.CuboidsGeneratorFromImgs(source=data_folder,
                                                             cub_frames=CUBOIDS_LENGTH,
                                                             prep_fn=resize_fn)
        data_test = istl.generators.ConsecutiveCuboidsGen(data_test)
    except Exception as e:
        print(f'Cannot load {data_folder}: {e}', file=sys.stderr)
        exit(-1)

    # Note the cumulative cuboids per each test video
    cum_cuboids_per_video = data_test.cum_cuboids_per_video

    try:
        test_labels = np.loadtxt(labels, dtype='int8')
    except Exception as e:
        print(f'Cannot load {labels}: ', str(e), file=sys.stderr)
        exit(-1)

    # Testing for each pair of anomaly and temporal values combinations
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
        meas['training_rec_error'] = dict(
            mean=float(sc_train.mean()),
            std=float(sc_train.std()),
            min=float(sc_train.min()),
            max=float(sc_train.max()),
        )

    meas['test_rec_error'] = dict(
        mean=float(true_scores_test.mean()),
        std=float(true_scores_test.std()),
        min=float(true_scores_test.min()),
        max=float(true_scores_test.max()),
    )

    meas['test_rec_error_norm'] = dict(
        mean=float(scores_test.mean()),
        std=float(scores_test.std()),
        min=float(scores_test.min()),
        max=float(scores_test.max()),
    )

    meas['results'] = istl.EvaluatorISTL._compute_perf_metrics(
        test_labels, pred,
        true_scores_test)

    # Separate prediction, scores and true labels per video ######
    scores_test_sep = split_measures_per_video(scores_test, cum_cuboids_per_video)
    pred_sep = split_measures_per_video(pred, cum_cuboids_per_video)
    test_labels_sep = split_measures_per_video(test_labels, cum_cuboids_per_video)

    for i in range(len(scores_test_sep)):
        # Indexes of predictions
        idxs = np.arange(pred_sep[i].size)

        # Compare ground-truth with true labels
        comp = np.zeros(pred_sep[i].size, dtype='uint8')

        # comp[pred_sep==0 && test_labels_sep==0] = 0 # TN
        comp[(pred_sep[i] == 1) & (test_labels_sep[i] == 1)] = 1  # TP
        comp[(pred_sep[i] == 0) & (test_labels_sep[i] == 1)] = 2  # FN
        comp[(pred_sep[i] == 1) & (test_labels_sep[i] == 0)] = 3  # FP

        groups = group_points_in_ranges(comp)
        true_labels = group_points_in_ranges(test_labels_sep[i])

        # True labels

        # Plot areas
        plt.figure(figsize=(12.8, 4.8))
        for gr in groups:
            if gr[0] == 0:
                continue
            plt.axvspan(ymin=0, ymax=1, xmin=gr[1][0], xmax=gr[1][1] + 0.5, alpha=0.3,
                        color='tab:orange' if gr[0] == 1 else 'tab:red' if gr[0] == 3 else 'tab:purple',
                        label='True Positive' if gr[0] == 1 else 'False Positive' if gr[0] == 3 else 'False Negative')

        # Plot lines
        for idx, tl in enumerate(true_labels):
            plt.plot(idxs[tl[1][0]: tl[1][1] + 1], scores_test_sep[i][tl[1][0]: tl[1][1] + 1],
                     label='Normal' if tl[0] == 0 else 'Abnormal',
                     color='tab:orange' if tl[0] == 1 else 'tab:blue')

            # Plot a line linking to the previous plotted line
            if idx < len(true_labels) - 1:
                plt.plot((tl[1][1], true_labels[idx + 1][1][0]),
                         (scores_test_sep[i][tl[1][1]], scores_test_sep[i][true_labels[idx + 1][1][0]]),
                         label='Normal' if tl[0] == 0 else 'Abnormal',
                         color='tab:orange' if tl[0] == 1 else 'tab:blue')

        plt.plot(idxs, np.repeat(anom_threshold, repeats=len(idxs)), '--', label='Anom threshold', color='tab:green')
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.title('Ground Truth')
        plt.xlabel('frames')
        plt.ylabel('Rec Error Norm')

        plt.savefig(output / f'test{i + 1}_ground_truth.pdf')
        plt.close()

    # Save the results
    with (output / 'measures.json').open('w') as f:
        json.dump(meas, f, indent=4)

    if not spatial_location:
        return

    # Perform spatial location
    localizator = istl.LocalizatorISTL(model=model,
                                       cub_frames=CUBOIDS_LENGTH,
                                       anom_thresh=anom_threshold,
                                       temp_thresh=temp_threshold,
                                       subwind_size=(SUBWIND_WIDTH,
                                                     SUBWIND_HEIGHT))
    anom_areas = localizator.spatial_loc_anomalies(data_test, pred, full_GPU=False)

    cub_idx = 0
    for i, pred_sep_i in enumerate(pred_sep):

        out_dir = output / f'-test{i + 1}'
        out_dir.mkdir(exist_ok=True)

        # Save all the frames
        for c, pred_sep_i_c in enumerate(pred_sep_i):

            cub = data_test[cub_idx + c]

            # Process all the frames wheter the last cuboid
            if c == len(pred_sep_i) - 1:
                fr = cub
            else:
                fr = [cub[0]]

            for j, frame in enumerate(map(lambda x: cvtColor(f, COLOR_GRAY2BGR), fr)):

                # Draw the anomalous subwindows if found
                if (cub_idx + c) in anom_areas:
                    recs = anom_areas[cub_idx + c]

                    for x, y in recs:
                        rectangle(frame, (x - (SUBWIND_WIDTH / 2), y + (SUBWIND_HEIGHT / 2)),
                                  (x + (SUBWIND_WIDTH / 2), y - (SUBWIND_HEIGHT / 2)),
                                  (255, 188, 0) if pred_sep_i_c == test_labels_sep[i][c] else (255, 0, 0), 2)

                        if pred_sep_i[c] == test_labels_sep[i][c]:
                            frame = addWeighted(frame, 0.4,
                                                SUBWIND_TP[y - (SUBWIND_WIDTH / 2): y + (SUBWIND_WIDTH / 2),
                                                x - (SUBWIND_WIDTH / 2): x + (SUBWIND_WIDTH / 2)], 0.5, 1)
                        else:
                            frame = addWeighted(frame, 0.4,
                                                SUBWIND_FP[y - (SUBWIND_WIDTH / 2): y + (SUBWIND_WIDTH / 2),
                                                x - (SUBWIND_WIDTH / 2): x + (SUBWIND_WIDTH / 2)], 0.5, 1)

                imwrite(str(out_dir / f'frame{c + j}.png'), cub)

        cub_idx += len(pred_sep_i)


if __name__ == '__main__':
    main()
