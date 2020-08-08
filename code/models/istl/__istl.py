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
import warnings
from copy import copy, deepcopy
from bisect import bisect_right
import numpy as np
#import tensorflow as tf
from tensorflow.keras import Model, Sequential, Input
from tensorflow.keras.layers import Conv2D, ConvLSTM2D, Conv2DTranspose
from tensorflow.keras.layers import Conv3D, Conv3DTranspose
from sklearn.metrics import roc_auc_score
from utils import confusion_matrix, equal_error_rate

def build_ISTL(cub_length: int):

	"""Builder function to construct an empty Tensorflow Keras Model holding
	the Incremental Spatio Temporal Learner (ISTL) architecture.

	Parameters
	----------


	"""

	if not isinstance(cub_length, int) or cub_length <= 0:
		raise ValueError('The cuboids length must be an integer greater than 0')

	istl = Sequential()

	"""
		The model receives grayscale frames of size 224 x 224 (only one channel).
	"""
	istl.add(Input(shape=(cub_length, 224, 224, 1)))

	"""
		C1: First Convolutional 2D layer.
		Kernel size: 27 x 27
		Filters: 128
		Strides: 4x4
	"""
	istl.add(Conv3D(filters=128, kernel_size=(1, 27, 27), strides=(1, 4, 4),
					name='C1', padding='same'))

	"""
		C2: Second Convolutional 2D layer.
		Kernel size: 13x13
		Filters: 64
		Strides: 2x2
	"""
	istl.add(Conv3D(filters=64, kernel_size=(1, 13, 13), strides=(1, 2, 2),
					name='C2', padding='same'))

	"""
		CL1: First Convolutional LSTM 2D layer
		Kernel size: 3x3
		Filters: 64
	"""
	istl.add(ConvLSTM2D(filters=64, kernel_size=(3,3), name='CL1', 
						return_sequences=True, padding='same'))

	"""
		CL2: Second Convolutional LSTM 2D layer
		Kernel size: 3x3
		Filters: 32
	"""
	istl.add(ConvLSTM2D(filters=32, kernel_size=(3,3), name='CL2', 
						return_sequences=True, padding='same'))

	"""
		DCL1: Third Convolutional LSTM 2D layer used for preparing deconvolution
		Kernel size: 3x3
		Filters: 64
	"""
	istl.add(ConvLSTM2D(filters=64, kernel_size=(3,3), name='DCL1', 
						return_sequences=True, padding='same'))

	"""
		DC1: First Deconvolution 2D Layer
		Kernel size: 13x13
		Filters: 64
		Strides: 2x2
	"""
	istl.add(Conv3DTranspose(filters=128, kernel_size=(1, 13, 13),
				strides=(1, 2, 2), name='DC1', padding='same'))

	"""
		DC2: Second Deconvolution 2D Layer
		Kernel size: 27x27
		Filters: 128
		Strides: 4x4
	"""
	istl.add(Conv3DTranspose(filters=1, kernel_size=(1, 27, 27),
				strides=(1, 4, 4), name='DC2', padding='same'))

	return istl

class ScorerLSTM:

	"""Handler class for the cuboids scoring through a previous trained
		ISTL Model

		Attributes
		----------

		model : tf.keras.Model
			Keras Model containing a pre-trained ISTL model

		cub_frames : int
			Number of frames conforming the cuboids
	"""

	def __init__(self, model: Model, cub_frames: int):

		# Check input
		if not isinstance(model, Model):
			raise ValueError('"model" should be a Keras model containing a '\
							'pretrained ISTL model')

		if not isinstance(cub_frames, int) or cub_frames <= 0:
			raise ValueError('"cub_frames" must be an integer greater than 0')

		# Copy to the object atributes
		self.__model = model
		self.__cub_frames = cub_frames

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
		self.score_video = np.vectorize(self.score_cuboid)
		"""


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

	def score_cuboid(self, cuboid: np.ndarray) -> np.float64:

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
		if not isinstance(cuboid, np.ndarray):
			raise ValueError('"cuboids" must be a numpy array')

		"""
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
		"""

		return np.sqrt(np.sum((cuboid - self.__model.predict(cuboid))**2))

	def score_cuboids(self, cub_set: np.array or list or tuple):

		"""Returns the reconstruction error of several input cuboids provided
			through a generator or a an array or list of cuboids

			Parameters
			----------

			cub_set: indexable and length-known collection of cuboids.
				(array, list or tuple of cuboids, generator of cuboids)
				Array, list, tuple or any generator of cuboids to be scored


			Raise
			-----

			ValueError: cub_set is not indexable or its length cannot be known
			ValueError: Any cuboid is not a valid numpy ndarray

			Return: 64-bit float numpy array vector with the reconstruction
					error of each collection's cuboid.
		"""

		# Check input
		if not hasattr(cub_set, '__getitem__') or not hasattr(cub_set,'__len__'):
			raise ValueError('Input cuboid\'s collection must have '\
								'__getitem__ and __len__ methods')

		# Procedure
		ret = np.zeros(len(cub_set), dtype='float64')

		for i in range(len(cub_set)):
			ret[i] = self.score_cuboid(cub_set[i])

		return ret
		
		#return np.apply_along_axis(self.score_cuboid, axis=0, )

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

	def predict_cuboid(self, cuboid: np.ndarray) -> bool:

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

	def predict_cuboids(self, cub_set, return_scores=False) -> np.ndarray:

		"""For each cuboid retrievable from a cuboid's collection (i.e.
			array-like object of cuboids or any genereator of cuboids), predicts
			wheter the cuboid is anomalous or not (i.e. its represents an
			anormal event or a normal event).

			A cuboid is considered anomalous when its reconstruction error is
			higher than the anomaly threshold and there's a number of anomalous
			consecutive cuboids greater than the temporal threshold

			Parameters
			----------
				cub_set: indexable and length-known collection of cuboids.
				(array, list or tuple of cuboids, generator of cuboids)
				Array, list, tuple or any generator of cuboids to be scored

				return_scores : bool (default False)
					Return the score associated to each cuboids or not.

					If true the function will return a tuple containing the
					vector with the predictions and the vector with the scores,
					if false, only the prediction vector is returned


			Raise
			-----

			ValueError: cub_set is not indexable or its length cannot be known
			ValueError: Any cuboid is not a valid numpy ndarray

			Return: 8-bit int numpy array vector with the prediction
					for each collection's cuboid if return_scores is False
					or a tuple with the prediction vector and the 64-bit float
					numpy array vector containing the reconstruction error
					associated to each cuboid.
		"""

		# Get the reconstruction error of cuboid's collection
		score = self.score_cuboids(cub_set)

		# Get cuboids whose reconstruction error overpass the anomalous threshold
		anom_cub = score >= self.__anom_thresh

		preds = np.zeros(len(cub_set), dtype='int8')

		# Count if there's temp_thresh consecutive anomalous cuboids
		cons_count = 0
		i_start = None
		for i in range(len(anom_cub)):

			if not anom_cub[i]:
				if cons_count >= self.__temp_thresh:
					preds[i_start: i] = 1

				cons_count = 0
				i_start = None

			else:

				if i_start is None:
					i_start = i

				cons_count += 1

		# Note the remainds cuboids
		if cons_count and cons_count >= self.__temp_thresh:
			preds[i_start:] = 1

		return preds if not return_scores else (preds, score)

	def predict_video(self, video: np.array) -> bool:

		# Check input
		if not isinstance(cuboids, np.ndarray):
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

	## Constructor ##
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
		return (np.array(self.__fp_cuboids) if len(self.__fp_cuboids) > 1 else
									np.expand_dims(self.__fp_cuboids, axis=0))

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
		pred, scores = self.predict_cuboids(cuboids, return_scores=True)
		"""
		scores = np.array(
		[self.score_cuboid(cuboids[i]) for i in range(len(cuboids))]).squeeze()
		

		pred = scores >= self.anom_thresh
		"""
		# Detect false positives and store them
		for i in range(pred.size):

			if pred[i] == 1 and labels[i] == 0:
				idx = bisect_right(self.__fp_rec_errors, scores[i])

				# Locate the index where the false positive cuboid must
				# be inserted to keep cuboids ordered from greater to lower
				# reconstruction error

				aux = cuboids[i] if cuboids[i].shape[0] != 1 else cuboids[i][0]

				self.__fp_rec_errors.insert(idx, scores[i])
				self.__fp_cuboids.insert(idx, aux)

				# Remove extra cuboids if max number of cuboid if specified
				if (self.__max_cuboids and len(self.__fp_cuboids) >=
														self.__max_cuboids):
					self.__fp_rec_errors = self.__fp_rec_errors[:self.__max_cuboids + 1]
					self.__fp_cuboids = self.__fp_cuboids[:self.__max_cuboids + 1]

		# Compute performance metrics
		cm = confusion_matrix(labels, pred)
		
		try:
			auc = roc_auc_score(labels, scores)
		except Exception as e:
			auc = np.NaN
			warnings.warn(str(e))

		try:
			eer = equal_error_rate(labels, scores)[0]
		except Exception as e:
			eer = np.NaN
			warnings.warn(str(e))

		return {
				'accuracy': float((cm[1, 1] + cm[0, 0])/len(scores)),
				'precission': float(cm[1, 1] / (cm[1, 1] + cm[0, 1])),
				'recall': float(cm[1, 1] / (cm[1, 1] + cm[1, 0])),
				'specificity': float(cm[0, 0] / (cm[0, 0] + cm[0, 1])),
				'AUC': float(auc),
				'EER': eer,
				'confusion matrix': {
					'TP': int(cm[1, 1]),
					'TN': int(cm[0, 0]),
					'FP': int(cm[0, 1]),
					'FN': int(cm[1, 0])
				}
		}

	def clear(self):

		"""Clears all the false positive cuboids stored
		"""

		self.__fp_cuboids = []
		self.__fp_rec_errors = []
