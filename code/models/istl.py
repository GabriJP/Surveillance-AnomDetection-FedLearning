# -*- coding: utf-8 -*-
###############################################################################
# Author: Nicolás Cubero Torres
# Description: Several utilities for the deployment and training of an
#			Incremental Spatio Temporal Learner model (ISTL). The ISTL model
#			is an unsupervised deep-learning approach for surveillance anomaly
#			detection that learns to reconstruct cuboids of video-frames
#			representing the normal behaviour with the lower loss, anomalies
#			are detected by the greater reconstruction error given by videos
#			which contains anomalies.
#
#			The ISTL model utilizes active learning with fuzzy aggregation,
#			to continuously update its knowledge of new anomalies and normality
#			that evolve over time.
#
# References:
#
# R. Nawaratne, D. Alahakoon, D. De Silva and X. Yu, "Spatiotemporal Anomaly
# Detection Using Deep Learning for Real-Time Video Surveillance," in
# IEEE Transactions on Industrial Informatics, vol. 16, no. 1, pp. 393-402,
# Jan. 2020, doi: 10.1109/TII.2019.2938527
###############################################################################

# Imported modules
import os
import warnings
import random
from copy import copy, deepcopy
from bisect import bisect_right
import numpy as np
#import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Conv2D, ConvLSTM2D, Conv2DTranspose
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from utils import make_partitions, confusion_matrix

def build_ISTL():

	"""Builder function to construct an empty Tensorflow Keras Model holding
	the Incremental Spatio Temporal Learner (ISTL) architecture.

	Parameters
	----------


	"""

	istl = Sequential()

	"""
		The model receives grayscale frames of size 224 x 224 (only one channel).
	"""
	istl.add(Input(shape=(224, 224)))

	"""
		C1: First Convolutional 2D layer.
		Kernel size: 27 x 27
		Filters: 128
		Strides: 4x4
	"""
	istl.add(Conv2D(filters=128, kernel_size=(27, 27), strides=4, name='C1'))

	"""
		C2: Second Convolutional 2D layer.
		Kernel size: 13x13
		Filters: 64
		Strides: 2x2
	"""
	istl.add(Conv2D(filters=64, kernel_size=(13, 13), strides=2, name='C2'))

	"""
		CL1: First Convolutional LSTM 2D layer
		Kernel size: 3x3
		Filters: 64
	"""
	istl.add(ConvLSTM2D(filters=64, kernel_size=(3,3)), name='CL1')

	"""
		CL2: Second Convolutional LSTM 2D layer
		Kernel size: 3x3
		Filters: 32
	"""
	istl.add(ConvLSTM2D(filters=32, kernel_size=(3,3)), name='CL2')

	"""
		DCL1: Third Convolutional LSTM 2D layer used for preparing deconvolution
		Kernel size: 3x3
		Filters: 64
	"""
	istl.add(ConvLSTM2D(filters=64, kernel_size=(3,3)), name='DCL1')

	"""
		DC1: First Deconvolution 2D Layer
		Kernel size: 13x13
		Filters: 64
		Strides: 2x2
	"""
	istl.add(Conv2DTranspose(filters=64, kernel_size=(13,13), strides=2, name='DC1'))

	"""
		DC2: Second Deconvolution 2D Layer
		Kernel size: 27x27
		Filters: 128
		Strides: 4x4
	"""
	istl.add(Conv2DTranspose(filters=128, kernel_size=(27, 27), strides=4, name='DC2'))

	return istl

class ScorerLSTM:

	"""Handler class for the cuboids scoring through a previous trained
		ISTL Model

		Atributes
		----------

		model : tf.keras.Model
			Keras Model containing a pre-trained ISTL model

		cub_frames : int
			Number of frames conforming the cuboids
	"""

	def __init__(self, model: Model, cub_frames: int):

		# Check input
		if not isistance(model, Model):
			raise ValueError('"model" should be a Keras model containing a '\
							'pretrained ISTL model')

		if not isinstance(cub_frames, int) or cub_frames <= 0:
			raise ValueError('"cub_frames" must be an integer greater than 0')

		# Copy to the object atributes
		self.__model = model
		self.__cub_frames = cub_frames

	## Observers ##

	@property
	def model(self):
		return self.__model

	@property
	def cub_frames(self):
		return self.__cub_frames

	## Setters ##

	@cub_frames.setter
	def cub_frames(self, value: int):

		if not isinstance(value, int) or value <= 0:
			raise ValueError('"cub_frames" must be an integer greater than 0')

		self.__cub_frames = value

	def score_cuboid(self, cuboid: np.array) -> np.float64:

		"""Returns the reconstruction error of an input cuboid

			Parameters
			----------

			cuboid: numpy array
				Array storing the cuboid of frames to be evaluated
				Dims of array must be (# of cuboids frames or lower, width, height)


			Raise
			-----

			ValueError: cuboid is not a numpy array
			ValueError: cuboid is not a 3-dimensional array of
						shape (?, width, height)

			Return: 64-bit float Numpy value
		"""

		# Check input
		if not isinstance(cuboids, np.array):
			raise ValueError('"cuboids" must be a numpy array')

		if cuboids.ndim != 3 or cuboids.shape[1:] != tuple(self.__model.input.shape[1:]):
			raise ValueError('cuboids must be a 3-dimensional array of '\
								'dims ({},{},{})'.format('?',
													self.__model.input.shape[1],
													self.__model.input.shape[2])
							)

		cuboid = cuboid[min(cuboid.shape[0], self.__cub_frames)]

		# Compute the sum of squared differences between the original
		#  and reconstructed frames
		rec_error = np.apply_along_axis(lambda frame: ((frame -
									self.__model.predict(frame))**2).sum(), 0)
		rec_error = np.square(rec_error.sum())

		return rec_error

	"""Returns the reconstruction error of each video's cuboids

		Parameters
		----------

		cuboid: numpy array
			Array storing the video frames which will be organized as a cuboids
			for its detection
			Dims of array must be (# frames, width, height)


		Raise
		-----

		ValueError: cuboid is not a numpy array
		ValueError: cuboid is not a 3-dimensional array of
					shape (?, width, height)

		Return: 64-bit float Numpy array with reconstruction error of
			each video cuboid
	"""
	score_video = np.vectorize(score_cuboid)

class PredictorLSTM(ScorerLSTM):

	"""Handler class for evaluation and prediction by a previous ISTL Model
		trained

		Attributes
		----------

		model : tf.keras.Model
			Keras Model containing a pre-trained ISTL model

		cub_frames : int
			Number of frames conforming the cuboids

		anom_thresh : float
			Anomaly Threshold for wich, a input cuboid is considered anomalous
			if its reconstruction error exceed it

		temp_thresh : int
			Number of consecutive cuboids classified as a anomalous (e.g. its
			reconstruction error exceed the anomalous threshold) required to
			consider a segment as anomalous
	"""

	def __init__(self, model: Model, cub_frames: int, anom_thresh: float,
															temp_thresh: int):

		super(PredictorLSTM, self).__init__(model, cub_frames)

		# Check input
		if (not isinstance(anom_thresh, (float, int)) or anom_thresh < 0 or
															anom_thresh > 1):
			raise ValueError('"anom_thresh" must be a float in [0, 1]')

		if not isinstance(temp_thresh, int) or temp_thresh < 0:
			raise ValueError('"temp_thresh" must be an integer greater or'\
							' equal than 0')


		# Copy to the object atributes
		self.__anom_thresh = anom_thresh
		self.__temp_thresh = temp_thresh

	## Observers ##
	@property
	def anom_thresh(self):
		return self.__anom_thresh

	@property
	def temp_thresh(self):
		return self.__temp_thresh

	## Setter ##
	@anom_thresh.setter
	def anom_thresh(self, value: float):

		if (not isinstance(value, (float, int)) or value < 0 or value > 1):
			raise ValueError('"anom_thresh" must be a float in [0, 1]')

		self.__anom_thresh = value

	@temp_thresh.setter
	def temp_thresh(self, value: int):

		if not isinstance(value, int) or value < 0:
			raise ValueError('"temp_thresh" must be an integer greater or'\
							' equal than 0')

		self.__temp_thresh = value

	def predict_cuboid(self, cuboid: np.array) -> bool:

		"""Predict wheter a cuboid is considered anomalous (i.e. its
			reconstruction error is greater than the anomalous threshold)

			Parameters
			----------

			cuboid: numpy array
				Array storing the cuboid of frames to be evaluated
				Dims of array must be (# of cuboids frames or lower, width, height)


			Raise
			-----

			ValueError: cuboid is not a numpy array
			ValueError: cuboid is not a 3-dimensional array of
						shape (?, width, height)

			Return: bool, True if anomaly, False otherwise
		"""

		return self.score_cuboid(cuboid) > self.__anom_thresh

	def predict_video(self, video: np.array) -> bool:

		# Check input
		if not isinstance(cuboids, np.array):
			raise ValueError('"cuboids" must be a numpy array')

		if cuboids.ndim != 3 or cuboids.shape[1:] != tuple(self.model.input.shape[1:]):
			raise ValueError('cuboids must be a 3-dimensional array of '\
								'dims ({},{},{})'.format('?',
													self.model.input.shape[1],
													self.model.input.shape[2])
							)

		# Evaluate the reconstruction error of each cuboid
		rec_errors = self.score_video(
						video[i, i + min(self.cub_frames,
							video.shape[0]%self.cub_frames)]
							for i in range(0, video.shape[0], self.cub_frames)
						)

		# Tag anomaly cuboids
		tags = rec_errors > self.__anom_thresh

		# Count if there's temp_thresh consecutive anomalous cuboids
		cons_count = 0
		for t in tags:

			if t > 0:
				cons_count += t
			else:
				cons_count = 0

			if cons_count >= self.__temp_thresh:
				return True

		return False

class EvaluatorLSTM(PredictorLSTM):

	"""Handler class for the performance evaluation of a LSTM model prediction.
		The utility stores the false positives cuboids classified by the trained
		model (the cuboids classified as normal by the model but anomalous by the
		user)

		Attributes
		----------

		model : tf.keras.Model
			Keras Model containing a pre-trained ISTL model

		cub_frames : int
			Number of frames conforming the cuboids

		anom_thresh : float
			Anomaly Threshold for wich, a input cuboid is considered anomalous
			if its reconstruction error exceed it

		temp_thresh : int
			Number of consecutive cuboids classified as a anomalous (e.g. its
			reconstruction error exceed the anomalous threshold) required to
			consider a segment as anomalous

		max_cuboids : int (Defalut: None)
			Max number of false positive cuboids to be stored. None for no
			limits.
	"""

	def __init__(self, model: Model, cub_frames: int, anom_thresh: float,
										temp_thresh: int, max_cuboids: int=None):
		super(EvaluatorLSTM, self).__init__(model, cub_frames, anom_thresh,
																	temp_thresh)

		# Private attributes
		self.__fp_cuboids = []	  # List of false positive stored cuboid
		self.__fp_rec_errors = [] # Reconstruction error of stored cuboid

		self.__max_cuboids = max_cuboids

	## Getters ##
	@property
	def fp_cuboids(self):
		return tuple(self.__fp_cuboids)

	def __len__(self):
		return len(self.__fp_cuboids)

	## Methods ##
	def evaluate_cuboids(self, cuboids: np.ndarray, labels: list or np.ndarray):

		"""Evaluates the prediction of the trained LSTM model given the anomaly
			threshold and temporal threshold provided. Aditionaly, all the false
			positive classified are stored

			Parameters
			----------

			cuboids : collection (list, tuple, numpy array) of cuboids
				Collection of cuboids to be classified

			labels : collection of integers
				Labels indicating 1 for anomaly or 0 for normal
		"""

		# Check input
		if not hasattr(cuboids, '__getitem__') or not hasattr(cuboids, '__len__'):
			raise ValueError('The passed cuboids are not a valid collection'\
																' of cuboids')

		if not hasattr(labels, '__getitem__') or not hasattr(labels, '__len__'):
			raise ValueError('The passed labels is not a valid collection of integers')

		if len(cuboids) != len(labels):
			raise ValueError('There must be a label for each cuboid')

		# Procedure
		if not isinstance(labels, np.ndarray):
			labels = np.array(labels)

		# Predict all cuboids
		scores = self.score_video(cuboids).squeeze()
		pred = scores >= self.anom_thresh

		# Detect false positives and store them
		for i in range(pred.size):

			if pred[i] == 1 and labels[i] == 0:
				idx = bisect_right(self.__fp_rec_errors, scores[i])

				# Locate the index where the false positive cuboid must
				# be inserted to keep cuboids ordered from greater to lower
				# reconstruction error

				self.__fp_rec_errors.insert(idx, scores[i])
				self.__fp_cuboids.insert(idx, cuboids[i])

				# Remove extra cuboids if max number of cuboid if specified
				if (self.__max_cuboids and len(self.__fp_cuboids) >=
														self.__max_cuboids):
					self.__fp_rec_errors = self.__fp_rec_errors[:self.__max_cuboids + 1]
					self.__fp_cuboids = self.__fp_cuboids[:self.__max_cuboids + 1]

		# Compute performance metrics
		cm = confusion_matrix(labels, pred)

		return {
				'accuracy': (cm[1, 1] + cm[0, 0])/len(scores),
				'precission': cm[1, 1] / (cm[1, 1] + cm[1, 0]),
				'recall': cm[1, 1] / (cm[1, 1] + cm[0, 1]),
				'specificity': cm[0, 0] / (cm[0, 0] + cm[1, 0]),
				'confusion matrix': {
					'TP': cm[1, 1],
					'TN': cm[0, 0],
					'FP': cm[1, 0],
					'FN': cm[0, 1]
				}
		}

	def clear(self):

		"""Clears all the false positive cuboids stored
		"""

		self.__fp_cuboids = []
		self.__fp_rec_errors = []

class CuboidsGenerator(Sequence):

	"""Data generator for the retrieval of cuboids from video files from
		a directory.

		The generator flows cuboids from videos files contained on a
		directory as they are accessed

		Parameters
		----------

		source: str
			Directory containing the videos to be flowed as cuboids

		cub_frames: int
			Number of frames contained on a cuboid

		prep_fn: function (optional)
			Function containing the preprocessing operations workflow to be
				applied to each video frame

		max_frames: int (default 100000)
			Max number of frames to be retrieved from disk when the generator
			flows cuboids from the videos
	"""

	def __init__(self, source: str, cub_frames: int, prep_fn=None,
							max_frames: int=1000, shuffle=False, seed=None):

		# Check input
		if not isinstance(source, str) or not source:
			raise ValueError('The source must be a valid directory path')

		if not isinstance(cub_frames, int) or cub_frames <= 0:
			raise ValueError('"cub_frames" must be grater than 0')

		if prep_fn and not callable(prep_fn):
			raise ValueError('"prep_fn" is not callable')

		if not isinstance(max_frames, int) or max_frames <= 0:
			raise ValueError('"max_frames" not a valid integer greater than 0')

		if not isinstance(shuffle, bool):
			raise ValueError('"shuffle" must be boolean')

		if seed is not None and not isinstance(seed, int):
			raise ValueError('"seed" must be None or integer')

		# Private attributes
		self.__source = source
		self.__cub_frames = cub_frames
		self.__prep_fn = prep_fn
		self.__max_frames = max_frames

		self.__total_cuboids = 0 # Number of total cuboids retrievable
		self.__total_frames = 0	 # Total frames retrievable from the videos
		self.__video_info = dict()

		self.__cuboids = None 	# Cuboids loaded from directory
		self.__loaded_cub_range = [None, None] # Range of cuboids loaded

		# Scan video files from directory
		if not os.path.isdir(self.__source):
			raise ValueError('"{}" not a valid directory'.format(self.__source))

		for f in sorted(os.listdir(self.__source)):

			dirname = (self.__source +
						('/' if not self.__source.endswith('/') else '') + f)

			# Check the listed directory is valid
			if not os.path.isdir(dirname):
				warnings.warn('"{}" not a directory with frames and will'\
												' be omitted'.format(dirname))
				continue

			"""
			cap = VideoCapture(f)

			if not cap.isOpened():
				warnings.warn('"{}" not a valid video file'.format(f))
				continue
			"""
			# Note the video attributes
			self.__video_info[dirname] = {
								'total_frames': len(os.listdir(dirname)),
								'num_cuboids': (int(len(os.listdir(dirname))/
														self.__cub_frames))
							}

		self.__video_fnames = list(self.__video_info.keys())

		# Shuffle the videos by the specified seed
		if shuffle:

			# Change the seed is specified
			if seed is not None:
				or_rand_state = random.getstate()
				random.seed(seed)

			random.shuffle(self.__video_fnames)

			# Recover the original random state
			if seed is not None:
				random.setstate(or_rand_state)

		 # Update progression and the countage if frames and cuboids
		self.__update_contage_data()

	def __update_contage_data(self):

		# Compute the total cuboids and total frames
		self.__total_frames = 0
		self.__total_cuboids = 0

		for v in self.__video_fnames:
			self.__total_frames += self.__video_info[v]['total_frames']
			self.__total_cuboids += self.__video_info[v]['num_cuboids']

		# Set a cuboid number mark for each video to let the sorted
		# access to cuboids
		self.__prog = np.zeros(len(self.__video_info), dtype=np.int32)

		for i in range(1, len(self.__prog)):
			self.__prog[i] = (self.__prog[i-1] +
				self.__video_info[self.__video_fnames[i-1]]['num_cuboids'])

	def __load_cuboids_video(self, dirname: str) -> np.ndarray:

		"""Loads a video stored as individual frames from directory as an
			array of cuboids.

			If the video cannot be agruped into cuboids of the specified length,
			last frame is repeated several frames until the agrupation can be
			performed.
		"""

		video = CuboidsGenerator.load_video(dirname, self.__prep_fn)

		# Count the extra number of frames needed to obtain complete cuboids
		extra_frames = video.shape[0] % self.__cub_frames

		if extra_frames:
			rep = self.__cub_frames - extra

			# Repeat the last frame of video until complete cuboids can be
			# formed through the video frames
			append = np.repeat(video[np.newaxis, -1], rep, axis=0)
			video = np.concatenate((video, append), axis=0)

		video = video.reshape(video.shape[0]//self.__cub_frames,
											self.__cub_frames, *video.shape[1:])

		return video

	def load_video(dirname: str, prep_fn=None) -> np.ndarray:

		"""Load a video stored as individual frames from a directory and returns
			it as a numpy array containing all the frames

			Parameters
			----------

			dirname: str
				Directory containing the videos to be flowed as cuboids

			prep_fn: function (optional)
				Function containing the preprocessing operations workflow to be
					applied to each video frame

		"""

		# Check input
		if not isinstance(dirname, str) or not dirname:
			raise ValueError('dirname must be a valid directory path')

		if prep_fn and not callable(prep_fn):
			raise ValueError('"prep_fn" is not callable')

		# Procedure

		if not os.path.isdir(dirname):
			raise ValueError('"{}" not a valid directory'.format(dirname))

		frames = []

		for f in sorted(os.listdir(dirname)):

			filename = (dirname + ('/' if not dirname.endswith('/') else '')+ f)

			# Loads the frame and store it
			try:
				img = img_to_array(load_img(filename))
			except Exception as e:
				print('Error with file {} which is going'\
								' to be ommited:"{}"'.format(filename, str(e)))
				continue

			# Apply preprocessing function if specified
			if prep_fn:
				img = prep_fn(img)

			# Check loaded images have the same format
			if frames and (frames[-1].shape != img.shape or
												frames[-1].dtype != img.dtype):
				raise ValueError('Differents sizes or types for images loaded'\
								' from "{}"'.format(dirname))

			frames.append(img)

		frames = np.array(frames)

		return frames

	def __len__(self) -> int:

		"""Returns the number of cuboids to be retrievable
		"""

		return self.__total_cuboids

	def __getitem__(self, idx) -> np.array:

		""" Retrieves the cuboid located at index
		"""

		# Check if desired cuboid is not loaded on RAM and retrieves
		# it from directory
		if self.__cuboids is None or not (idx >= self.__loaded_cub_range[0] and
											idx <= self.__loaded_cub_range[1]):

			frames_loaded = 0

			# Located the video containing the desired cuboid
			v_idx = np.searchsorted(self.__prog, idx, side='right')
			i = v_idx

			self.__loaded_cub_range[0] = idx
			self.__loaded_cub_range[1] = self.__loaded_cub_range[0]

			# Read videos until max_frames are loaded
			while True:

				if ((frames_loaded +
					self.__video_info[self.__video_fnames[i]]['total_frames']) >
					self.__max_frames or i >= len(self.__video_fnames)):
					break

				print('-',self.__video_fnames[i])

				# Load the video cuboids and append to the previous loaded ones
				if self.__cuboids is None:
					self.__cuboids = self.__load_cuboids_video(
														self.__video_fnames[i])
				else:
					self.__cuboids = np.concatenate((self.__cuboids,
							self.__load_cuboids_video(self.__video_fnames[i])),
							axis=0)

				# Count the loaded frames
				frames_loaded += self.__video_info[self.__video_fnames[i]][
																'total_frames']

				# Note the retrieved cuboids into the loaded cuboid range
				self.__loaded_cub_range[1] += self.__video_info[
									self.__video_fnames[i]]['num_cuboids'] - 1

				i += 1


		# The desired cuboid is actually stored on RAM and can be returned
		return self.__cuboids[idx - self.__loaded_cub_range[0]]

	def make_partitions(self, partitions: tuple, shuffle=False, seed=None):

		"""Returns several partitions of cuboids set as a tuple of Cuboids
			Generator holding the desired partitions

			Parameters
			----------

			partitions : tuple of float
				Ratios of video to be contained on each partition.
				The sum of ratios must be 1

			shuffle : boolean
				Shuffle the video before partitioning or not

			seed : int
				The seed to be used for shuffling

			Return
			------
			Tuple containing as many cuboids generator as desired partitions
		"""

		# Check input
		if not partitions:
			raise ValueError('Some portion value must be passed')

		if any(not isinstance(x, (float, int)) for x in partitions):
			raise ValueError('Portion values are not number')

		if np.abs(sum(partitions) - 1) > epsilon:
			raise ValueError('portion values doesn\'t sum 1')

		if not isinstance(shuffle, bool):
			raise ValueError('"shuffle" must be boolean')

		if seed is not None and not isinstance(seed, int):
			raise ValueError('"seed" must be None or integer')

		# Procedure

		video_fnames = copy(self.__video_fnames)

		# Shuffle the videos by the specified seed
		if shuffle:

			# Change the seed is specified
			if seed is not None:
				or_rand_state = random.getstate()
				random.seed(seed)

			random.shuffle(video_fnames)

			# Recover the original random state
			if seed is not None:
				random.setstate(or_rand_state)

		# Partitionate videos and create cuboid generator for each partition
		video_part = make_partitions(video_fnames, *partitions)
		cubgens = []

		for p in video_part:

			# Create new cuboid generator
			cubgens.append(deepcopy(self))

			cubgens[-1].__cuboids = None
			cubgens[-1].__loaded_cub_range = [None, None]
			cubgens[-1].__video_info.clear()
			cubgens[-1].__video_fnames = p

			# Copy the selected video info on the new generator
			for v in cubgens[-1].__video_fnames:
				cubgens[-1].__video_info[v] = self.__video_info[v]

			# Update progression and total frames and cuboids data
			cubgens[-1].__update_contage_data()

		return tuple(cubgens)

class FramesFromCuboidsGen(Sequence):

	"""Data generator for the flowing of individuals frames from a set
		of cuboids stored as an array or list or through a Cuboid Generator

		Parameters
		----------

		cub_frames : int
			Number of frames contained on each cuboid

		args : collection (list, tuple, CuboidsGenerator) of Cuboids
			Only one collection can be passed containing the cuboids to be
			presented as an individual frames

			 : Many numpy arrays
			Many numpy arrays representing individual cuboids
	"""

	def __init__(self, cub_frames: int, *args: list or tuple or np.ndarray or CuboidsGenerator):

		# Check input
		if not isinstance(cub_frames, int) or cub_frames <= 0:
			raise ValueError('"cub_frames" must be an integer greater than 0')

		if len(args) > 1  and any(not isinstance(x, np.ndarray) or
											len(x) != cub_frames for x in args):
			raise ValueError('Any cuboid provided not a valid numpy array or'\
									' not have the number of frames specified')
		elif (not hasattr(args[0], '__getitem__') or
											not hasattr(args[0], '__len__')):
			raise ValueError('cuboids collection passed must support an '\
												'item getter and len operator')

		# Copy to private attributes
		if len(args) == 1:
			args = args[0]

		self.__cuboids = args
		self.__cub_frames = cub_frames

	def __len__(self):

		"""Returns the total number of frames to be retrievable
		"""
		return len(self.__cuboids)*self.__cub_frames

	def __getitem__(self, idx):

		"""Returns the frame located at the desired index

			Parameters
			----------

			idx : int
				Index of frame to be retrieved

			Return
			------
				numpy ndarray representing the desired frame
		"""
		return self.__cuboids[idx//self.__cub_frames][idx%self.__cub_frames]
