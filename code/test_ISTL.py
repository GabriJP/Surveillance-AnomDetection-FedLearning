# -*- coding: utf-8 -*-
"""

@author: Nicolás Cubero Torres
@description: Test the prediction of an ISTL model for a given test video.

@usage: test_ISTL.py -m <Pretrained model's h5 file>
					-t <Path to the file or folder containing the test video>
"""
# Modules imported
import sys
import os
import argparse
import numpy as np
from cv2 import imread, imwrite, resize, cvtColor, COLOR_BGR2GRAY
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
from models import istl

#from tensorflow.keras import backend as K

# Constants
CUBOIDS_LENGTH = 8
CUBOIDS_WIDTH = 224
CUBOIDS_HEIGHT = 224

# Image resize function
resize_fn = lambda img: np.expand_dims(resize(cvtColor(img, COLOR_BGR2GRAY),
						(CUBOIDS_WIDTH, CUBOIDS_HEIGHT))/255, axis=2)

def get_single_test(filename: str, prep_fn):

	sz = 200
	test = [] #np.zeros(shape=(sz, 256, 256, 1))
	#cnt = 0

	for f in sorted(os.listdir(filename)):
		if str(os.path.join(filename, f))[-3:] == "tif":
			img = imread(os.path.join(filename, f))
			img = prep_fn(img)

			test.append(img)
			#cnt = cnt + 1
	return np.array(test)

def make_cuboids(test_video: np.ndarray, cub_length: int,
				cub_width: int, cub_height: int):
	sz = test.shape[0] - cub_length
	sequences = np.zeros((sz, cub_length, cub_width, cub_height, 1))

	# apply the sliding window technique to get the sequences
	for i in range(0, sz):
		clip = np.zeros((cub_length, cub_width, cub_height, 1))

		for j in range(cub_length):
			clip[j] = test[i + j, :, :, :]

		sequences[i] = clip

	return sequences

### Input Arguments
parser = argparse.ArgumentParser(description='Test an Incremental Spatio'\
							' Temporal Learner model for a given test video')
parser.add_argument('-m', '--model', help='A pretrained model stored on a'\
					' h5 file', type=str)
parser.add_argument('-t', '--test_video', help='Path to folder or file'\
					' containing the test video', type=str)
parser.add_argument('-o', '--output', help='Output file in which the results'\
					' will be located', type=str)
parser.add_argument('-s', '--save_reconstruction', help='Save the reconstructed'\
					' cuboids',
					action='store_true', default=False)

args = parser.parse_args()

model_fn = args.model
test_video_path = args.test_video
output = args.output
save_rec = args.save_reconstruction

dot_pos = output.rfind('.')
if dot_pos != -1:
	results_fn = output[:dot_pos] + '.svg'
else:
	results_fn = output + '.svg'

### Loads model
try:
	model = load_model(model_fn)
except Exception as e:
	print('Cannot load the model: ', str(e), file=sys.stderr)
	exit(-1)

### Load the video test
try:
	test = get_single_test(test_video_path, resize_fn)
except Exception as e:
	print('Cannot load {}: '.format(test_video_path), str(e), file=sys.stderr)
	exit(-1)

data_train = istl.generators.CuboidsGeneratorFromImgs(
								source='/home/ncubero/UCSD_Anomaly_Dataset.'\
										'v1p2/UCSDped1/Train/',
								cub_frames=CUBOIDS_LENGTH,
								prep_fn=resize_fn)

# Make cuboids
cub_test = make_cuboids(test, CUBOIDS_LENGTH, CUBOIDS_WIDTH, CUBOIDS_HEIGHT)

### Predicts cuboid
evaluator = istl.ScorerISTL(model, CUBOIDS_LENGTH)
evaluator.fit(data_train)
scores = evaluator.score_cuboids(np.expand_dims(cub_test, axis=1), True)

#scores = (scores - scores.min()) / scores.max()

### Plot results
plt.plot(scores)
plt.ylabel('Anomaly score')
plt.xlabel('frame t')
plt.savefig(results_fn)

if save_rec:
	rec = np.zeros((len(cub_test), CUBOIDS_WIDTH, CUBOIDS_HEIGHT, 1), dtype=np.float64)

	# Reconstruct all frames
	for c in range(len(cub_test)):
		rec[c,:,:,0] = model.predict(cub_test[c:c+1])[0,0,:,:,0]

	rec = (rec.squeeze()*255).astype('int32')

	# Save the reconstructed frames
	if not os.path.isdir(output):
		try:
			os.mkdir(output)
		except Exception as e:
			print('Cannot save the reconstructed frames: ', str(e))
			exit(-1)

	for c in range(len(rec)):
		imwrite(output + '/' + 'frame{}.png'.format(c), rec[c])
