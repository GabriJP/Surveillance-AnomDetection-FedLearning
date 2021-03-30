# -*- coding: utf-8 -*-
"""

@author: Nicolás Cubero Torres
@description: Test the prediction of an ISTL model for a given test video.

@usage: test_ISTL.py -m <Pretrained model's h5 file>
                     -t <Path to the file or folder containing the test video>
"""
# Modules imported
import argparse
import imghdr
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
from cv2 import imread, imwrite, resize, cvtColor, COLOR_BGR2GRAY
from tensorflow import config
from tensorflow.keras.models import load_model

from models import istl
from utils import root_sum_squared_error

physical_devices = config.experimental.list_physical_devices('GPU')
config.experimental.set_memory_growth(physical_devices[0], True)

# Constants
CUBOIDS_LENGTH = 8
CUBOIDS_WIDTH = 224
CUBOIDS_HEIGHT = 224

# Image resize function
resize_fn = lambda img: np.expand_dims(resize(cvtColor(img, COLOR_BGR2GRAY),
                                              (CUBOIDS_WIDTH, CUBOIDS_HEIGHT)) / 255, axis=2)

# Interval format regex
interval_format = re.compile('[0-9]+-[0-9]+')


def get_single_test(dirname: str, prep_fn):
    test = None  # np.zeros(shape=(sz, 256, 256, 1))
    # f_frame = None
    cnt = 0

    listdir = sorted(os.listdir(dirname))

    for d in listdir:

        filename = os.path.join(dirname, d)

        if imghdr.what(filename):
            img = imread(filename)
            img = prep_fn(img)

            if test is None:
                test = np.zeros((len(listdir), *img.shape))

            test[cnt] = img
            cnt = cnt + 1

    return test


def make_cuboids(test_video: np.ndarray, cub_length: int,
                 cub_width: int, cub_height: int):
    sz = test_video.shape[0] - cub_length
    sequences = np.zeros((sz, cub_length, cub_width, cub_height, 1))

    # apply the sliding window technique to get the sequences
    for i in range(0, sz):
        # clip = np.zeros((cub_length, cub_width, cub_height, 1))

        for j in range(cub_length):
            sequences[i, j] = test_video[i + j, :, :, :]
    # clip[j] = test[i + j, :, :, :]

    # sequences[i] = clip

    return sequences


def make_frames_labels(n_frames: int, interval_frames_ranges: list):
    labels = np.zeros(n_frames, dtype='uint8')

    for inter in interval_frames_ranges:
        # Put the frames belonging to each interval as anomalous
        min_v, max_v = tuple(int(lim) for lim in inter.split('-'))
        labels[min_v: max_v] = 1

    return labels


def find_differential_frames(n_frames: int, interval_frames_ranges: list):
    diff_frames = [0]

    for inter in interval_frames_ranges:

        # Note the changing frames
        for limit in (int(lim) for lim in inter.split('-')):
            # if limit != diff_frames[-1]:
            diff_frames.append(limit)

    # Note the last frame
    if n_frames - 1 != diff_frames[-1]:
        diff_frames.append(n_frames - 1)

    return diff_frames


### Input Arguments
parser = argparse.ArgumentParser(description='Test an Incremental Spatio' \
                                             ' Temporal Learner model for a given test video')
parser.add_argument('-m', '--model', help='A pretrained model stored on a' \
                                          ' h5 file', type=str)
parser.add_argument('-t', '--test_video', help='Path to folder or file' \
                                               ' containing the test video', type=str)
parser.add_argument('-o', '--output', help='Output file in which the results' \
                                           ' will be located', type=str)
parser.add_argument('-i', '--interval', help='Anomaly frames intervals',
                    type=str, nargs='+')
parser.add_argument('-l', '--lower', help='Lower training score evaluated by' \
                                          ' the model', type=float, nargs='?')
parser.add_argument('-g', '--higher', help='Higher training score evaluated by' \
                                           ' the model', type=float, nargs='?')
parser.add_argument('-s', '--save_reconstruction', help='Save the reconstructed' \
                                                        ' cuboids',
                    action='store_true', default=False)

args = parser.parse_args()

model_fn = args.model
test_video_path = args.test_video
output = args.output
save_rec = args.save_reconstruction
interval = args.interval
lower = args.lower
higher = args.higher

# Check input
if any(not interval_format.match(inter) for inter in interval):
    print('Any interval specified not a valid interval with format "<start frame>-<stop frame>"', file=sys.stderr)
    exit(-1)

dot_pos = output.rfind('.')
if dot_pos != -1:
    results_fn = output[:dot_pos] + '.svg'
else:
    results_fn = output + '.svg'

# Loads model
try:
    model = load_model(model_fn, custom_objects=dict(root_sum_squared_error=root_sum_squared_error))
except Exception as e:
    print(f'Cannot load the model: {e}', file=sys.stderr)
    exit(-1)

# Load the video test
try:
    test = get_single_test(test_video_path, resize_fn)
except Exception as e:
    print(f'Cannot load {test_video_path}: {e}', file=sys.stderr)
    exit(-1)

# data_train = istl.generators.CuboidsGeneratorFromImgs(
#                                source='/home/ncubero/UCSD_Anomaly_Dataset.'\
#                                        'v1p2/UCSDped1/Train/',
#                                cub_frames=CUBOIDS_LENGTH,
#                                prep_fn=resize_fn)

# data_train = istl.generators.ConsecutiveCuboidsGen(data_train)

# Make cuboids
cub_test = make_cuboids(test, CUBOIDS_LENGTH, CUBOIDS_WIDTH, CUBOIDS_HEIGHT)

# Predicts cuboid
evaluator = istl.ScorerISTL(model, CUBOIDS_LENGTH)
# evaluator.fit(data_train)
scores = evaluator.score_cuboids(np.expand_dims(cub_test, axis=1), False)
if lower is None:
    scores = (scores - scores.min()) / scores.max()
else:
    scores = (scores - lower) / higher
index = np.arange(scores.shape[0])

# Plot results
inter = find_differential_frames(scores.shape[0], interval)
anom = False
for i in range(len(inter) - 1):
    plt.plot(index[inter[i]: inter[i + 1]],
             scores[inter[i]: inter[i + 1]],
             color='blue' if not anom else 'red')
    # label='Normal' if not anom else 'Abnormal')

    anom = not anom

plt.legend()
plt.ylabel('Anomaly score')
plt.xlabel('frame t')
plt.savefig(results_fn)

if save_rec:
    rec = np.zeros((len(cub_test), CUBOIDS_WIDTH, CUBOIDS_HEIGHT, 1), dtype=np.float64)

    # Reconstruct all frames
    for c in range(len(cub_test)):
        rec[c, :, :, 0] = model.predict(cub_test[c:c + 1])[0, 0, :, :, 0]

    rec = (rec.squeeze() * 255).astype('int32')

    # Save the reconstructed frames
    if not os.path.isdir(output):
        try:
            os.mkdir(output)
        except Exception as e:
            print(f'Cannot save the reconstructed frames: {e}')
            exit(-1)

    for c in range(len(rec)):
        imwrite(output + '/' + f'frame{c}.png', rec[c])
