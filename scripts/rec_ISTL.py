# -*- coding: utf-8 -*-
"""

@author: Nicolás Cubero Torres
@description: Reconstruct a video from a pre-trained ISTL model.

@usage: rec_ISTL.py -m <Pretrained model's h5 file>
                    -t <Path to the file or folder containing the test video>
"""
# Modules imported
import imghdr
import sys
from pathlib import Path

import click
import numpy as np
from cv2 import imread, imwrite, resize, cvtColor, COLOR_BGR2GRAY
from tensorflow import config
from tensorflow.keras.models import load_model

from utils import root_sum_squared_error

physical_devices = config.experimental.list_physical_devices('GPU')
config.experimental.set_memory_growth(physical_devices[0], True)

# Constants
CUBOIDS_LENGTH = 8
CUBOIDS_WIDTH = 224
CUBOIDS_HEIGHT = 224


# Image resize function
def resize_fn(img):
    np.expand_dims(resize(cvtColor(img, COLOR_BGR2GRAY), (CUBOIDS_WIDTH, CUBOIDS_HEIGHT)) / 255, axis=2)


def get_single_test(dirname: str, prep_fn):
    test = None  # np.zeros(shape=(sz, 256, 256, 1))
    # f_frame = None
    cnt = 0

    files = sorted(Path(dirname).iterdir())
    for filename in files:

        if imghdr.what(filename):
            img = imread(filename)
            img = prep_fn(img)

            if test is None:
                test = np.zeros((len(files), *img.shape))

            test[cnt] = img
            cnt += 1

    return test[:cnt]


def make_cuboids(test_video: np.ndarray, cub_length: int,
                 cub_width: int, cub_height: int):
    sz = test_video.shape[0] - cub_length + 1
    sequences = np.zeros((sz, cub_length, cub_width, cub_height, 1))

    # apply the sliding window technique to get the sequences
    for i in range(0, sz):
        # clip = np.zeros((cub_length, cub_width, cub_height, 1))

        for j in range(cub_length):
            sequences[i, j] = test_video[i + j, :, :, :]
    # clip[j] = test[i + j, :, :, :]

    # sequences[i] = clip

    return sequences


@click.command()
@click.option('-m', '--model', help='A pretrained model stored on a h5 file', type=str)
@click.option('-t', '--test_video', help='Path to folder or file containing the test video', type=str)
@click.option('-o', '--output', help='Output file in which the results will be located', type=str)
def main(model, test_video, output):
    """Reconstruct a test video from a pretrained ISTL model"""
    # Loads model
    try:
        model = load_model(model, custom_objects=dict(root_sum_squared_error=root_sum_squared_error))
    except Exception as e:
        print('Cannot load the model: ', str(e), file=sys.stderr)
        return

        # Load the video test
    try:
        test = get_single_test(test_video, resize_fn)
    except Exception as e:
        print(f'Cannot load {test_video}: {e}', file=sys.stderr)
        return

        # Make cuboids
    cub_test = make_cuboids(test, CUBOIDS_LENGTH, CUBOIDS_WIDTH, CUBOIDS_HEIGHT)

    rec = np.zeros((len(test), CUBOIDS_WIDTH, CUBOIDS_HEIGHT, 1), dtype=np.float64)

    # Reconstruct all frames
    for c in range(len(cub_test)):
        pred = model.predict(cub_test[c:c + 1])
        rec[c, :, :, 0] = pred[0, 0, :, :, 0]

        # For last frame, all the remain frames are stacked
        if c == len(cub_test) - 1:
            for i in range(1, CUBOIDS_LENGTH):
                rec[c + i, :, :, 0] = pred[0, i, :, :, 0]

    rec = (rec.squeeze() * 255).astype('int32')

    # Save the reconstructed frames
    output = Path(output)
    output.mkdir(exist_ok=True)

    for c in range(len(rec)):
        imwrite(str(output / f'{c:03d}.png'), rec[c])


if __name__ == '__main__':
    main()
