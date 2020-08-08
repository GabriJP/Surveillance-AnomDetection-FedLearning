# -*- coding: utf-8 -*-
###############################################################################
# Author: Nicolás Cubero Torres
# Description: Several utilities for the retrieval of video data from datasets
#				as cuboids or frames supported by the Incremental Spatio
#				Temporal Learner architecture
#
# References:
#
# R. Nawaratne, D. Alahakoon, D. De Silva and X. Yu, "Spatiotemporal Anomaly
# Detection Using Deep Learning for Real-Time Video Surveillance," in
# IEEE Transactions on Industrial Informatics, vol. 16, no. 1, pp. 393-402,
# Jan. 2020, doi: 10.1109/TII.2019.2938527
###############################################################################

# Imported modules
from sys import float_info
import os
import warnings
import random
from copy import copy, deepcopy
import imghdr
import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from utils import make_partitions

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

		max_cuboids: int (default 100)
			Max number of consecutive cuboids to be retrieved from disk when
			the generator flows cuboids from the videos
	"""

	def __init__(self, source: str, cub_frames: int, prep_fn=None,
					batch_size=None, max_cuboids: int=100, shuffle=False,
					seed=None, return_cub_as_label=False):

		# Check input
		if not isinstance(source, str) or not source:
			raise ValueError('The source must be a valid directory path')

		if not isinstance(cub_frames, int) or cub_frames <= 0:
			raise ValueError('"cub_frames" must be grater than 0')

		if prep_fn and not callable(prep_fn):
			raise ValueError('"prep_fn" is not callable')

		if (batch_size is not None and (not isinstance(batch_size, int) or
															batch_size <= 0)):
			raise ValueError('"batch_size" must be greater than 0')

		if not isinstance(max_cuboids, int) or max_cuboids <= 0:
			raise ValueError('"max_cuboids" not a valid integer greater than 0')

		if not isinstance(shuffle, bool):
			raise ValueError('"shuffle" must be boolean')

		if seed is not None and not isinstance(seed, int):
			raise ValueError('"seed" must be None or integer')

		if not isinstance(return_cub_as_label, bool):
			raise ValueError('"return_cub_as_label" must be boolean')

		if batch_size and max_cuboids < batch_size:
			raise ValueError('The batch size cannot be greater than the '\
																'max cuboids')

		# Private attributes
		self.__source = source
		self.__cub_frames = cub_frames
		self.__prep_fn = prep_fn
		self.__batch_size = batch_size
		self.__max_cuboids = max_cuboids
		self.__return_cub_as_label = return_cub_as_label

		self.__cuboids_info = [] # Path file of cuboids
		#self.__video_info = dict()

		self.__cuboids = None 	# Cuboids loaded from directory
		self.__loaded_cub_range = [None, None] # Range of cuboids loaded

		# Scan video files from directory
		if not os.path.isdir(self.__source):
			raise ValueError('"{}" not a valid directory'.format(self.__source))

		self.__scan_source_dir()

		# To store the order in which cuboids will be accessed
		self.__access_cuboids = copy(self.__cuboids_info)

		# Shuffle the cuboids
		if shuffle:
			self.shuffle(True, seed)


	def __scan_source_dir(self):

		"""Scans the desired video directory looking for all video frames files
			and note them into cuboids for posterior loading
		"""

		self.__cuboids_info = []

		for d in sorted(os.listdir(self.__source)):

			dirname = (self.__source +
						('/' if not self.__source.endswith('/') else '') + d)

			# Check the listed directory is valid
			if not os.path.isdir(dirname):
				warnings.warn('"{}" not a directory with frames and will'\
												' be omitted'.format(dirname))
				continue


			# List frames and group them into cuboids
			cub_info = (dirname, [])
			for f in sorted(os.listdir(dirname)):

				frame_fname = dirname+'/'+f

				# Note the frame to cuboid if is a valid image file
				if imghdr.what(frame_fname):
					cub_info[1].append(f)

					if len(cub_info[1]) == self.__cub_frames:
						self.__cuboids_info.append(cub_info)
						cub_info = (dirname, [])

			# Add the remaind cuboid and fill it by repetitions of last frame
			# until make the cuboid with the derired number of frames
			if len(cub_info[1]) > 0:
				rest = self.__cub_frames - len(cub_info[1])
				cub_info[1].extend([cub_info[1][-1]]*rest)

				self.__cuboids_info.append(cub_info)

	def __load_cuboid(filenames: list or tuple, prep_fn=None):

		"""Loads a cuboid from its video frames files
		"""
		
		frames = []

		for fn in filenames:

			# Loads the frame and store it
			try:
				img = img_to_array(load_img(fn))
			except:
				print(fn)
				raise

			# Apply preprocessing function if specified
			if prep_fn:
				img = prep_fn(img)

			# Check loaded images have the same format
			if frames and (frames[-1].shape != img.shape or
												frames[-1].dtype != img.dtype):
				raise ValueError('Differents sizes or types for images loaded'\
								' detected for image "{}"'.format(fn))

			frames.append(img)

		frames = np.array(frames)

		return frames

	### Observers

	@property
	def return_cub_as_label(self):
		return self.__return_cub_as_label

	@property
	def batch_size(self):
		return self.__batch_size

	def __len__(self) -> int:

		"""Returns the number of cuboids to be retrievable or batches of cuboids
			if batch_size where specified
		"""

		return int(len(self.__cuboids_info) if self.__batch_size is None else
									len(self.__cuboids_info)//self.__batch_size)

	def __getitem__(self, idx) -> np.ndarray:

		""" Retrieves the cuboid located at index or the cuboids batch wheter
			batch_size has been provided
		"""
		cub_idx = self.__batch_size * idx if self.__batch_size else idx

		# Check if desired cuboid is not loaded on RAM and retrieves
		# it from directory
		if (self.__cuboids is None or not
									(cub_idx >= self.__loaded_cub_range[0] and
									cub_idx < self.__loaded_cub_range[1])):


			self.__loaded_cub_range[0] = cub_idx
			self.__loaded_cub_range[1] = min(cub_idx + self.__max_cuboids,
											len(self.__cuboids_info))


			self.__cuboids = np.array([
				CuboidsGenerator.__load_cuboid(
					(self.__access_cuboids[i][0]+'/'+
						fp for fp in self.__access_cuboids[i][1]),
					self.__prep_fn) for i in range(
						self.__loaded_cub_range[0], self.__loaded_cub_range[1])
				])


		# The desired cuboid/vatch is actually stored on RAM and can be returned
		start = cub_idx - self.__loaded_cub_range[0]
		end = start + (self.__batch_size if self.__batch_size else 1)

		#	If only one cuboid is returned, the cuboid number dimension
		#	cannot be supresed
		ret = self.__cuboids[start:end]

		if self.__return_cub_as_label:
			ret = (ret, ret)

		return ret

	### Modifiers
	@return_cub_as_label.setter
	def return_cub_as_label(self, v):
		if not isinstance(v, bool):
			raise ValueError('"return_cub_as_label" must be bool')

		self.__return_cub_as_label = v

	@batch_size.setter
	def batch_size(self, v):

		if (v is not None and (not isinstance(v, int) or v <= 0)):
			raise ValueError('batch_size must be greater than 0')

		self.__batch_size = v

	def shuffle(self, shuf=False, seed=None):

		"""Shuffle randomly the cuboids or undo the shufflering making the
			cuboids to recover its original order

			Parameters
			----------

			shuf : bool
				True for shuffle the cuboids or False to make the cuboids
				recover the original order

			seed : int or None
				seed to be used for the random shufflering
		"""

		if not isinstance(shuf, bool):
			raise ValueError('"shuf" must be boolean')

		if seed is not None and not isinstance(seed, int):
			raise ValueError('"seed" must be None or integer')

		if shuf:

			# Change the seed is specified
			if seed is not None:
				or_rand_state = random.getstate()
				random.seed(seed)

			random.shuffle(self.__access_cuboids)

			# Recover the original random state
			if seed is not None:
				random.setstate(or_rand_state)

		else:
			self.__access_cuboids = copy(self.__cuboids_info)

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

		if np.abs(sum(partitions) - 1) > float_info.epsilon:
			raise ValueError('portion values doesn\'t sum 1')

		if not isinstance(shuffle, bool):
			raise ValueError('"shuffle" must be boolean')

		if seed is not None and not isinstance(seed, int):
			raise ValueError('"seed" must be None or integer')

		# Procedure

		# Partitionate cuboids and create cuboid generator for each partition
		cuboids_part = make_partitions(self.__access_cuboids, *partitions)
		cubgens = []

		for p in cuboids_part:

			# Create new cuboid generator
			cubgens.append(deepcopy(self))

			cubgens[-1].__cuboids = None
			cubgens[-1].__loaded_cub_range = [None, None]
			cubgens[-1].__cuboids_info = p
			cubgens[-1].__access_cuboids = copy(p)

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

	def __init__(self, cub_frames: int, *args: list or tuple or np.ndarray or
															CuboidsGenerator):

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
