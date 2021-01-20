# -*- coding: utf-8 -*-
"""

@author: Nicolás Cubero Torres
@description: Reconstruct a video from a pre-trained ISTL model.

@usage: rec_ISTL.py -m <Pretrained model's h5 file>
					-t <Path to the file or folder containing the test video>
"""
# Modules imported
import sys
import os
import argparse
import imghdr
import numpy as np
from cv2 import imread, imwrite, resize, cvtColor, COLOR_BGR2GRAY
from tensorflow.keras.models import load_model
from tensorflow import __version__ as tf_version
#from PIL import Image
#import matplotlib.pyplot as plt
from models import istl
from utils import root_sum_squared_error

if tf_version.startswith('1'):
	from tensorflow import ConfigProto, Session

	config = ConfigProto()
	config.gpu_options.allow_growth = True
	sess = Session(config=config)
else:
	from tensorflow import config

	physical_devices = config.experimental.list_physical_devices('GPU')
	config.experimental.set_memory_growth(physical_devices[0], True)


# Constants
CUBOIDS_LENGTH = 8
CUBOIDS_WIDTH = 224
CUBOIDS_HEIGHT = 224

# Image resize function
resize_fn = lambda img: np.expand_dims(resize(cvtColor(img, COLOR_BGR2GRAY),
						(CUBOIDS_WIDTH, CUBOIDS_HEIGHT))/255, axis=2)

def get_single_test(dirname: str, prep_fn):

	test = None #np.zeros(shape=(sz, 256, 256, 1))
	#f_frame = None
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
			cnt += 1

	return test[:cnt]

def make_cuboids(test_video: np.ndarray, cub_length: int,
				cub_width: int, cub_height: int):
	sz = test.shape[0] - cub_length + 1
	sequences = np.zeros((sz, cub_length, cub_width, cub_height, 1))

	# apply the sliding window technique to get the sequences
	for i in range(0, sz):
		#clip = np.zeros((cub_length, cub_width, cub_height, 1))

		for j in range(cub_length):
			sequences[i, j] = test[i + j, :, :, :]
			#clip[j] = test[i + j, :, :, :]

		#sequences[i] = clip

	return sequences

### Input Arguments
parser = argparse.ArgumentParser(description='Reconstruct a test video from a'\
							' pretrained ISTL model')
parser.add_argument('-m', '--model', help='A pretrained model stored on a'\
					' h5 file', type=str)
parser.add_argument('-t', '--test_video', help='Path to folder or file'\
					' containing the test video', type=str)
parser.add_argument('-o', '--output', help='Output file in which the results'\
					' will be located', type=str)

args = parser.parse_args()

model_fn = args.model
test_video_path = args.test_video
output = args.output

### Loads model
try:
	model = load_model(model_fn,custom_objects={'root_sum_squared_error':                   
							root_sum_squared_error})
except Exception as e:
	print('Cannot load the model: ', str(e), file=sys.stderr)
	exit(-1)

### Load the video test
try:
	test = get_single_test(test_video_path, resize_fn)
except Exception as e:
	print('Cannot load {}: '.format(test_video_path), str(e), file=sys.stderr)
	exit(-1)

# Make cuboids
cub_test = make_cuboids(test, CUBOIDS_LENGTH, CUBOIDS_WIDTH, CUBOIDS_HEIGHT)

rec = np.zeros((len(test), CUBOIDS_WIDTH, CUBOIDS_HEIGHT, 1), dtype=np.float64)

# Reconstruct all frames
for c in range(len(cub_test)):
	pred = model.predict(cub_test[c:c+1])
	rec[c,:,:,0] = pred[0,0,:,:,0]

	# For last frame, all the remain frames are stacked
	if c == len(cub_test) - 1:
		for i in range(1, CUBOIDS_LENGTH):
			rec[c + i,:,:,0] = pred[0,i,:,:,0]

rec = (rec.squeeze()*255).astype('int32')

# Save the reconstructed frames
if not os.path.isdir(output):
	try:
		os.mkdir(output)
	except Exception as e:
		print('Cannot save the reconstructed frames: ', str(e))
		exit(-1)

for c in range(len(rec)):
	imwrite(output + '/' + '{:03d}.png'.format(c), rec[c])
