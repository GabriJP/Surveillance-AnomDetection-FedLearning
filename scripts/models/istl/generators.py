# -*- coding: utf-8 -*-
###############################################################################
# Author: Nicolás Cubero Torres
# Description: Several utilities for the retrieval of video data from datasets
#                as cuboids or frames supported by the Incremental Spatio
#                Temporal Learner architecture
#
# References:
#
# R. Nawaratne, D. Alahakoon, D. De Silva and X. Yu, "Spatiotemporal Anomaly
# Detection Using Deep Learning for Real-Time Video Surveillance," in
# IEEE Transactions on Industrial Informatics, vol. 16, no. 1, pp. 393-402,
# Jan. 2020, doi: 10.1109/TII.2019.2938527
###############################################################################

# Imported modules
import imghdr
import os
import random
import warnings
from copy import copy, deepcopy
from sys import float_info
from typing import List, Tuple, Union

import numpy as np
from cv2 import VideoCapture
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import Sequence

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
                 batch_size=1, max_cuboids: int = 300, shuffle=False,
                 seed=None, return_cub_as_label=False):

        # Check input
        if not isinstance(source, str) or not source:
            raise ValueError('The source must be a valid directory path')

        if not isinstance(cub_frames, int) or cub_frames <= 0:
            raise ValueError('"cub_frames" must be grater than 0')

        if prep_fn and not callable(prep_fn):
            raise ValueError('"prep_fn" is not callable')

        if not isinstance(batch_size, int):
            raise TypeError('"batch_size" must be integer')

        if batch_size <= 0:
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
            raise ValueError('The batch size cannot be greater than the ' \
                             'max cuboids')

        # Private attributes
        self.__source = source
        self.__cub_frames = cub_frames
        self.__prep_fn = prep_fn
        self.__batch_size = batch_size
        self.__max_cuboids = max_cuboids
        self.__return_cub_as_label = return_cub_as_label
        self.__shuffle = False

        # Maximum number of batches to load from disk
        self.__max_batch = self.__max_cuboids // self.__batch_size

        self._cuboids_info = []  # Path file of cuboids
        self._video_info = []

        self._cuboids = None  # Cuboids loaded from directory
        self.__loaded_cub_range = [None, None]  # Range of cuboids loaded

        # Scan video files from directory
        if not os.path.isdir(self.__source):
            raise ValueError('"{}" not a valid directory'.format(self.__source))

        self._scan_source_dir()

        # To store the order in which cuboids will be accessed
        self._access_cuboids = copy(self._cuboids_info)

        # Shuffle the cuboids
        if shuffle:
            self.shuffle(True, seed)

    def _scan_source_dir(self):

        """Scans the desired video directory looking for all video files
            and note them into cuboids for posterior loading.

            @note This method should be reimplemented on the derived class
        """
        raise NotImplementedError()

    def _load_cuboid(self, cuboid: tuple, prep_fn=None):

        """Loads a cuboid from its video frames files

            @note This method should be reimplemented on the derived class
        """
        raise NotImplementedError()

    def _update_video_info(self):

        raise NotImplementedError()

    @property
    def return_cub_as_label(self):
        return self.__return_cub_as_label

    @property
    def batch_size(self):
        return self.__batch_size

    @property
    def source(self):
        return self.__source

    @property
    def cub_frames(self):
        return self.__cub_frames

    @property
    def prep_fn(self):
        return self.__prep_fn

    @property
    def max_cuboids(self):
        return self.__max_cuboids

    @property
    def video_info(self):
        return tuple(self._video_info)

    def is_shuffled(self):
        return self.__shuffle

    def __len__(self) -> int:

        """Returns the number of cuboids to be retrievable or batches of cuboids
            if batch_size where specified
        """

        return np.ceil(len(self._access_cuboids) / self.__batch_size).astype(int)

    def n__getitem__(self, idx) -> np.ndarray:

        """ Retrieves the cuboid located at index or the cuboids batch wheter
            batch_size has been provided
        """

        """
        if idx < 0 or idx >= len(self):
            raise IndexError('index out of range')

        cub_idx = self.__batch_size * idx

        # Check if desired cuboid is not loaded on RAM and retrieves
        # it from directory
        if (self._cuboids is None or not
                                    (cub_idx >= self.__loaded_cub_range[0] and
                                    cub_idx < self.__loaded_cub_range[1])):


            self.__loaded_cub_range[0] = cub_idx
            self.__loaded_cub_range[1] = min(cub_idx + self.__max_batch *
                                                        self.__batch_size,
                                            len(self._access_cuboids))


            self._cuboids = np.array([
                type(self)._load_cuboid(
                    self._access_cuboids[i],
                    self.__prep_fn) for i in range(
                        self.__loaded_cub_range[0], self.__loaded_cub_range[1])
                ])

            # Normalize cuboids
            #self._cuboids = (self._cuboids - self._cuboids.mean()) / self._cuboids.std()

        # The desired cuboid/vatch is actually stored on RAM and can be returned
        start = cub_idx - self.__loaded_cub_range[0]
        end = min(start + self.__batch_size, len(self._access_cuboids))

        #    If only one cuboid is returned, the cuboid number dimension
        #    cannot be supresed
        ret = self._cuboids[start:end]

        if self.__return_cub_as_label:
            ret = (ret, ret)

        return ret
        """
        return self.__getslice__(idx, idx + 1)

    def __getitem__(self, idx: int):

        """Retrieves the cuboids or cuboids' batch wheter batch_size is privided
            from index i to index j
        """
        if isinstance(idx, int):
            start, stop, step = idx, idx + 1, 1
        elif isinstance(idx, slice):
            start, stop, step = idx.start, idx.stop, idx.step
        else:
            raise TypeError('The passed index must be integer or an slice')

        if start >= len(self) or stop > len(self):
            raise IndexError('Index out of range')

        if start < 0:
            start %= len(self)

        if stop < 0:
            stop %= len(self)

        if (stop - start) > self.__max_batch:
            raise RuntimeError('Slice retrived overpass the max number of cuboids retrievable specified')

        start = self.__batch_size * start
        stop = self.__batch_size * stop

        # Check if desired cuboids are not loaded on RAM and retrieves
        # it from directory
        if (self._cuboids is None or not (
                start >= self.__loaded_cub_range[0] and start < self.__loaded_cub_range[1]) or not (
                stop >= self.__loaded_cub_range[0] and stop <= self.__loaded_cub_range[1])):
            self.__loaded_cub_range[0] = start
            self.__loaded_cub_range[1] = min(start + self.__max_batch *
                                             self.__batch_size,
                                             len(self._access_cuboids))

            self._cuboids = np.array([
                self._load_cuboid(
                    self._access_cuboids[i],
                    self.__prep_fn) for i in range(
                    self.__loaded_cub_range[0], self.__loaded_cub_range[1])
            ])

        # Normalize cuboids
        # self._cuboids = (self._cuboids - self._cuboids.mean()) / self._cuboids.std()

        # The desired cuboid/vatch is actually stored on RAM and can be returned
        start = start - self.__loaded_cub_range[0]
        stop = stop - self.__loaded_cub_range[0]

        #    If only one cuboid is returned, the cuboid number dimension
        #    cannot be supresed
        ret = self._cuboids[start:stop:step]

        if self.__return_cub_as_label:
            ret = (ret, ret)

        return ret

    @return_cub_as_label.setter
    def return_cub_as_label(self, v):
        if not isinstance(v, bool):
            raise ValueError('"return_cub_as_label" must be bool')

        self.__return_cub_as_label = v

    @batch_size.setter
    def batch_size(self, v):

        if not isinstance(v, int) or v <= 0:
            raise ValueError('batch_size must be greater than 0')

        if v and self.__max_cuboids < v:
            raise ValueError('The batch size cannot be greater than the max cuboids')

        self.__batch_size = v
        self._cuboids = None

        # Recompute the max batch retrievable
        self.__max_batch = self.__max_cuboids // self.__batch_size

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

        self.__shuffle = shuf

        if shuf:

            # Change the seed is specified
            if seed is not None:
                or_rand_state = random.getstate()
                random.seed(seed)

            random.shuffle(self._access_cuboids)

            # Recover the original random state
            if seed is not None:
                random.setstate(or_rand_state)

        else:
            self._access_cuboids = copy(self._cuboids_info)

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
        if not hasattr(partitions, '__getitem__'):
            raise TypeError('"partitions" must be a indexable collection of float or ints')

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
        cuboids_part = make_partitions(self._access_cuboids, *partitions)
        cubgens = []

        for p in cuboids_part:
            # Create new cuboid generator
            cubgens.append(deepcopy(self))

            cubgens[-1]._cuboids = None
            cubgens[-1].__loaded_cub_range = [None, None]
            cubgens[-1]._cuboids_info = p
            cubgens[-1]._access_cuboids = copy(p)
            cubgens[-1]._update_video_info()

        return tuple(cubgens)

    def take_subpartition(self, port: float, seed=None):

        """Split the current generator into two generators, containing the first
            one, a random subpartition of the current generator and the second
            one the remained cuboids

            Parameters:
            -----------
            port : float
                Portion of the original cuboids to select for the subpartition

            seed : int
                The seed to be used for the subpartition's random selection
        """

        # Check input
        if not isinstance(port, float):
            raise TypeError('The portion for the subpartition must be float')

        if port <= 0 or port >= 1:
            raise ValueError('The portion for the subpartition must be in (0, 1)')

        if seed is not None and not isinstance(seed, int):
            raise TypeError('The seed must be None or integer')

        if seed is not None and seed <= 0:
            raise ValueError('The seed must be greater than 0')

        # Procedure
        cubgens = tuple(deepcopy(self) for i in range(2))
        cubgens[0]._cuboids_info = []

        part_size = int(port * len(self._access_cuboids))
        sel_index = set()

        # Change the seed is specified
        if seed is not None:
            or_rand_state = random.getstate()
            random.seed(seed)

        # Select a random subpartition
        for i in range(part_size):

            idx = None

            while idx is None or idx in sel_index:
                idx = random.randint(0, len(cubgens[1]._cuboids_info) - 1)

            cubgens[0]._cuboids_info.append(self._cuboids_info[idx])
            del cubgens[1]._cuboids_info[idx]

            sel_index.add(idx)

        # Recover the original random state
        if seed is not None:
            random.setstate(or_rand_state)

        # Update the partition's video information
        for cub in cubgens:
            cub._cuboids = None
            cub.__loaded_cub_range = [None, None]
            cub._access_cuboids = cub._cuboids_info
            cub._update_video_info()

        return cubgens

    def take_cons_subpartition(self, port: float):

        """Split the current generator into two generators, containing the first
            one, a subpartition containing the given porcentage of last video
            cuboids and the second one the remained ones.

            Parameters:
            -----------
            port : float
                Portion of the original cuboids to select for the subpartition
        """

        # Check input
        if not isinstance(port, float):
            raise TypeError('The portion for the subpartition must be float')

        if port <= 0 or port >= 1:
            raise ValueError('The portion for the subpartition must be in (0, 1)')

        # Procedure
        cubgens = tuple(deepcopy(self) for i in range(2))
        cubgens[0]._cuboids_info = []

        # Select a random subpartition
        for vid_inf in self._video_info:

            # Determine the number of cuboids to extract from current video
            # and the current index to get
            part_size = np.round(port * vid_inf['num_cuboids']).astype(int)

            if not part_size:
                break

            idx = vid_inf['first_cuboid_index']

            # Add the selected cuboids if choosen to the subpartition and remove
            # from the second one
            cubgens[0]._cuboids_info.extend(self._cuboids_info[idx: idx + part_size])
            del cubgens[1]._cuboids_info[idx: idx + part_size]

        # Update the partition's video information
        for cub in cubgens:
            cub._cuboids = None
            cub.__loaded_cub_range = [None, None]
            cub._access_cuboids = cub._cuboids_info
            cub._update_video_info()

        return cubgens

    def merge(*args):

        """Merge several Cuboids Generators into one single Cuboids Generator

            Parameters:
            -----------

            args : Several Cuboids Generators of the same type
        """
        # Check input

        if not args:
            raise ValueError('Any Cuboid Generator must be provided')

        for i in range(len(args)):
            if not isinstance(args[i], CuboidsGenerator):
                raise TypeError('A passed generator not a valid cuboid generator')

            if i > 0 and type(args[i]) is not type(args[i - 1]):
                raise TypeError('All the generators must below to the same type' \
                                ' of Cuboid Generator')

        # Procedure
        new = deepcopy(args[0])

        new._cuboids = None
        new.__loaded_cub_range = [None, None]
        new._cuboids_info.extend([cub for cg in args[1:] for cub in cg._access_cuboids])
        new._access_cuboids = copy(new._cuboids_info)
        new._update_video_info()

        return new

    def augment_data(self, rate=None, **kwargs):

        """Augment the retrievable by transformations applied to the original
            cuboids wheter the rate is larger than 0. if rate is 0, disable the
            data augmentation

            Parameters
            ----------

            rate: float (optional)
                Max ratio of augmented cuboids to generate. The number of augmented
                cuboids will be (rate * # original cuboids) which will be
                added among the original cuboids. It will be greater than 0

            Note: Function not implemented
        """
        raise NotImplementedError()

    @property
    def cum_cuboids_per_video(self):
        cub_per_vid = list(np.ceil(v['frames'] / self.__cub_frames)
                           for v in self._video_info)

        for i in range(1, len(cub_per_vid)):
            cub_per_vid[i] += cub_per_vid[i - 1]

        return tuple(cub_per_vid)


class CuboidsGeneratorFromImgs(CuboidsGenerator):
    """Implementation of the Cuboids Generator utility for the retrieval of
        cuboids from a video directory conforming one folder for each video
        containing the video frames as individual images.

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
                 batch_size=1, max_cuboids: int = 300, shuffle=False,
                 seed=None, return_cub_as_label=False):

        super(CuboidsGeneratorFromImgs, self).__init__(source, cub_frames,
                                                       prep_fn, batch_size,
                                                       max_cuboids, shuffle,
                                                       seed,
                                                       return_cub_as_label)

    def _scan_source_dir(self):

        """Scans the desired video directory looking for all video frames files
            and note them into cuboids for posterior loading
        """

        self._cuboids_info = []

        for d in sorted(os.listdir(self.source)):

            dirname = (self.source + ('/' if not self.source.endswith('/') else '') + d)

            # Check the listed directory is valid
            if not os.path.isdir(dirname):
                warnings.warn(f'"{dirname}" not a directory with frames and will be omitted')
                continue

            # List frames and group them into cuboids
            cub_info = (dirname, [])
            for f in sorted(os.listdir(dirname)):

                frame_fname = dirname + '/' + f

                # Note the frame to cuboid if is a valid image file
                if imghdr.what(frame_fname):
                    cub_info[1].append(f)

                    if len(cub_info[1]) == self.cub_frames:
                        self._cuboids_info.append(cub_info)
                        cub_info = (dirname, [])

            # Add the remaind cuboid and fill it by repetitions of last frame
            # until make the cuboid with the derired number of frames
            if len(cub_info[1]) > 0:
                rest = self.cub_frames - len(cub_info[1])
                cub_info[1].extend([cub_info[1][-1]] * rest)

                self._cuboids_info.append(cub_info)

        # Note the video's information
        self._update_video_info()

    def _load_cuboid(self, cuboid: tuple, prep_fn=None):

        """Loads a cuboid from its video frames files
        """

        filenames = (cuboid[0] + '/' + fp for fp in cuboid[1])

        frames = []

        for fn in filenames:

            # Loads the frame and store it
            try:
                img = img_to_array(load_img(fn))
            except Exception as e:
                print(f'{fn}: {e}')
                raise

            # Apply preprocessing function if specified
            if prep_fn:
                img = prep_fn(img)

            # Check loaded images have the same format
            if frames and (frames[-1].shape != img.shape or
                           frames[-1].dtype != img.dtype):
                raise ValueError(f'Differents sizes or types for images loaded detected for image "{fn}"')

            frames.append(img)

        frames = np.array(frames)

        return frames

    def _update_video_info(self):

        video_info_list = []

        video_info = None
        for i, cub_info in enumerate(self._cuboids_info):

            if not video_info or video_info['video fname'] != cub_info[0]:
                video_info = {'video fname': cub_info[0], 'frames': 0,
                              'first_cuboid_index': i, 'num_cuboids': 0}
                video_info_list.append(video_info)

            video_info['frames'] += len(set(cub_info[1]))
            video_info['num_cuboids'] += 1

        self._video_info = video_info_list

    def augment_data(self, rate=None, **kwargs):

        """Augment the retrievable by transformations applied to the original
            cuboids wheter the rate is larger than 0. if rate is 0, disable the
            data augmentation

            Parameters
            ----------

            rate: float (optional)
                Max ratio of augmented cuboids to generate. The number of augmented
                cuboids will be (rate * # original cuboids) which will be
                added among the original cuboids. It will be greater than 0

            max_stride: int
                Max stride applied between the original frames to construct the
                generated cuboids. It will be in (0,# video frames/cuboids length]
        """

        # check input
        if rate is not None and not isinstance(rate, (float, int)):
            raise TypeError('"rate" must be float or int')

        if rate is not None and rate <= 0.0:
            raise ValueError('"rate" must be greater or equal than 0')

        if 'max_stride' not in kwargs:
            raise AttributeError('"max_stride" must be provided')

        max_stride = kwargs['max_stride']

        if not isinstance(max_stride, int):
            raise TypeError('"max_stride" must be int')

        if max_stride < 2:
            raise ValueError('"max_stride" must be greater than 1')

        # Procedure
        n_aug_cuboids = int(rate * len(self._access_cuboids)) if rate else -1

        if n_aug_cuboids == 0 and rate > 0.0:
            warnings.warn('Rate specified is too low and no cuboid will be augmented')

        ori_cub_idx = 0  # Current video index used for generate a cuboid
        stride = 2
        start_frame_idx = 0  # First video frame to be added to the generated cuboid
        new_cuboids = []  # New cuboids generated
        augment = True

        frame_idx = start_frame_idx

        i = 0
        while i != n_aug_cuboids:

            new_cuboid = (self._cuboids_info[ori_cub_idx][0], [])

            # Forming a new cuboid from a strided-frames windows
            for j in range(self.cub_frames):
                new_cuboid[1].append(self._cuboids_info[ori_cub_idx][1][frame_idx])
                frame_idx += stride

                if frame_idx >= len(self._cuboids_info[ori_cub_idx][1]):
                    # Next frames must be taken from the following cuboid
                    ori_cub_idx += 1

                if ori_cub_idx == len(self._cuboids_info):
                    # No more cuboids can be generated from the original
                    # cuboids so let's start again from the start with a higher
                    # stride
                    start_frame_idx += 1
                    frame_idx = start_frame_idx
                    ori_cub_idx = 0
                else:
                    frame_idx %= len(self._cuboids_info[ori_cub_idx][1])

                if start_frame_idx == stride:
                    # A next video is reached so is not posible to generate
                    # the cuboid
                    start_frame_idx = 0
                    frame_idx = start_frame_idx
                    stride += 1

                if stride > max_stride:
                    # No more stride can be used so let's start for the first
                    # stride value using a new start frame index
                    augment = False
                    break

                if self._cuboids_info[ori_cub_idx][0] != new_cuboid[0]:
                    # A next video is reached so is not posible to generate
                    # the cuboid

                    break

            if len(new_cuboid[1]) == self.cub_frames:
                # Check if the cuboid is formed among all the required frames
                new_cuboids.append(new_cuboid)

            i += 1

            if not augment:
                if n_aug_cuboids != -1:
                    warnings.warn('Only {} new cuboids could be ' \
                                  'generated with the configuration' \
                                  ' provided'.format(i - 1))
                break

        # Agregate the new cuboids generated to the info and update object
        self._cuboids_info.extend(new_cuboids)
        self._access_cuboids = copy(self._cuboids_info)
        self._update_video_info()


class CuboidsGeneratorFromVid(CuboidsGenerator):
    """Implementation of the Cuboids Generator utility for the retrieval of
        cuboids from a video directory containing a video file for each video.

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
                 batch_size=1, max_cuboids: int = 300, shuffle=False,
                 seed=None, return_cub_as_label=False):

        super(CuboidsGeneratorFromVid, self).__init__(source, cub_frames,
                                                      prep_fn, batch_size,
                                                      max_cuboids, shuffle,
                                                      seed,
                                                      return_cub_as_label)

        self.__cap_opened = None  # To make more efficient the video retrieval

    def _scan_source_dir(self):

        """Scans the desired video directory looking for all video files
            and note them into cuboids for posterior loading
        """

        self._cuboids_info = []

        for d in sorted(os.listdir(self.source)):

            fname = (self.source + ('/' if not self.source.endswith('/') else '') + d)

            fv = VideoCapture(fname)

            # Get video file and split its frames in cuboids
            if not fv.isOpened():
                warnings.warn(f'"{fname}" not a valid video file and will be omitted')
                continue

            n_frames = int(fv.get(7))

            # Note the information's video
            # self._video_info.append({'video fname': fname,
            # 'frames': n_frames})

            # List frames and group them into cuboids
            for i in range(0, n_frames, self.cub_frames):
                self._cuboids_info.append((fname, i,
                                           min(i + self.cub_frames, n_frames) - 1,
                                           1))

            fv.release()

        # Note the information's video
        self._update_video_info()

    def _update_video_info(self):

        video_info_list = []

        video_info = None
        for i, cub_info in enumerate(self._cuboids_info):

            if not video_info or video_info['video fname'] != cub_info[0]:
                video_info = {'video fname': cub_info[0], 'frames': 0,
                              'first_cuboid_index': i, 'num_cuboids': 0}
                video_info_list.append(video_info)

            video_info['num_cuboids'] += 1
            video_info['frames'] += np.floor((cub_info[2] - cub_info[1] + 1) / cub_info[3]).astype(
                int)  # len(set(cub_info[1]))

        self._video_info = video_info_list

    def _load_cuboid(self, cuboid: tuple, prep_fn=None):

        """Loads a cuboid from its video frames files
        """
        fname = cuboid[0]
        frames_range = cuboid[1:-1]
        stride = cuboid[-1]

        cuboid_data = None
        n_frames = frames_range[0]  # Number of readen frames

        # Load the desired video frames
        if (self.is_shuffled() or self.__cap_opened is None or
                self.__cap_opened[0] != fname or
                frames_range[0] != self.__cap_opened[2] + 1):

            if self.__cap_opened is not None:
                self.__cap_opened[1].release()

            vid = VideoCapture(fname)

            if not vid.isOpened():
                raise ValueError(f'Cannot load frames {frames_range} from video file {fname}')

            vid.set(1, frames_range[0])

            # Store the Video Capture for retrieval of next cuboid belonging
            # to the same video
            if not self.is_shuffled():
                self.__cap_opened = [fname, vid, frames_range[1]]
        else:
            vid = self.__cap_opened[1]
            self.__cap_opened[2] = frames_range[1]  # Update the last frame taken

        for i in range(self.cub_frames):

            if n_frames <= frames_range[1]:  # readen

                # Take consecutive frames separated by the stride
                ret, fr = vid.read()

                if not ret:
                    raise ValueError('Failed to load frame {} from video ' \
                                     'file {}'.format(i, fname))

                n_frames += 1

                # Apply preprocessing function if specified
                if prep_fn:
                    fr = prep_fn(fr)

                # Add the frame readen to the cuboid
                if cuboid_data is None:
                    cuboid_data = np.zeros((self.cub_frames,
                                            *fr.shape), dtype=fr.dtype)

                cuboid_data[i] = fr

                # Exclude the next middle frames if stride
                for _ in range(stride - 1):
                    if n_frames <= frames_range[1]:
                        vid.grab()
                        n_frames += 1

            else:

                # Repeat the last frame if a complete cuboid cannot be taken
                # from the remained video frames
                cuboid_data[i] = cuboid_data[i - 1]

        return cuboid_data

    def augment_data(self, rate=None, **kwargs):

        """Augment the retrievable by transformations applied to the original
            cuboids wheter the rate is larger than 0. if rate is 0, disable the
            data augmentation

            Parameters
            ----------

            rate: float (optional)
                Max ratio of augmented cuboids to generate. The number of augmented
                cuboids will be (rate * # original cuboids) which will be
                added among the original cuboids. It will be greater than 0

            max_stride: int
                Max stride applied between the original frames to construct the
                generated cuboids. It will be in (0,# video frames/cuboids length]
        """

        # check input
        if rate is not None and not isinstance(rate, (float, int)):
            raise TypeError('"rate" must be float or int')

        if rate is not None and rate <= 0.0:
            raise ValueError('"rate" must be greater or equal than 0')

        if 'max_stride' not in kwargs:
            raise AttributeError('"max_stride" must be provided')

        max_stride = kwargs['max_stride']

        if not isinstance(max_stride, int):
            raise TypeError('"max_stride" must be int')

        if max_stride < 2:
            raise ValueError('"max_stride" must be greater than 1')

        # Procedure
        n_aug_cuboids = int(rate * len(self._access_cuboids)) if rate else -1

        if n_aug_cuboids == 0 and rate > 0.0:
            warnings.warn('Rate specified is too low and no cuboid will be augmented')

        ori_vid_idx = 0  # Current video index used for generate a cuboid
        ori_cub_idx = 0  # First cuboid of the current video
        stride = 2
        start_frame_idx = 0  # First video frame to be added to the generated cuboid
        new_cuboids = []  # New cuboids generated
        new_cuboid = None
        augment = True

        frame_idx = start_frame_idx + self._cuboids_info[ori_cub_idx][1]

        i = 0
        while i != n_aug_cuboids:

            # Construct a new cuboid
            new_cuboid = (self._video_info[ori_vid_idx]['video fname'],
                          frame_idx,
                          min(frame_idx + self.cub_frames * stride,
                              self._video_info[ori_vid_idx]['frames'] +
                              self._cuboids_info[ori_cub_idx][1]) - 1,
                          stride)

            # Prepare the following cuboid
            frame_idx = new_cuboid[2] + 1

            # Discard non-valid cuboids
            if (new_cuboid[2] - new_cuboid[1] + 1) >= new_cuboid[3]:
                new_cuboids.append(new_cuboid)

            if frame_idx - self._cuboids_info[ori_cub_idx][1] == self._video_info[ori_vid_idx]['frames']:
                # Next frames must be taken from the following cuboid
                ori_cub_idx += np.ceil(self._video_info[ori_vid_idx]['frames'] / self.cub_frames).astype(int)
                ori_vid_idx += 1
                frame_idx = start_frame_idx

            if ori_vid_idx == len(self._video_info):
                # No more cuboids can be generated from the original
                # cuboids so let's start again from the start with a higher
                # stride
                start_frame_idx += 1
                ori_vid_idx = 0
                ori_cub_idx = 0
                frame_idx = start_frame_idx + self._cuboids_info[ori_cub_idx][1]

            if start_frame_idx == stride:
                # A next video is reached so is not posible to generate
                # the cuboid
                start_frame_idx = 0
                frame_idx = start_frame_idx + self._cuboids_info[ori_cub_idx][1]
                stride += 1

            if stride > max_stride:
                # No more stride can be used so let's start for the first
                # stride value using a new start frame index
                break

            i += 1

        if n_aug_cuboids != -1 and i < n_aug_cuboids:
            warnings.warn(f'Only {i - 1} new cuboids could be generated with the configuration provided')

        # Agregate the new cuboids generated to the info and update object
        self._cuboids_info.extend(new_cuboids)
        self._access_cuboids = copy(self._cuboids_info)
        self._update_video_info()

    def __del__(self):
        if self.__cap_opened is not None:
            self.__cap_opened[1].release()


# super(CuboidsGeneratorFromVid, self).__del__()


class ConsecutiveCuboidsGen(Sequence):
    """Data generator for the retrieval of cuboids made from each posible
        original video's set of consecutive frames

        Attributes
        ----------

        cub_gen : CuboidsGenerator instance
            Original generator of cuboids from which the cuboids will be
            retrieved
    """

    def __init__(self, cub_gen: CuboidsGenerator):

        # Check input
        if not isinstance(cub_gen, CuboidsGenerator):
            raise TypeError('A valid cuboid\'s generator must be provided')

        # Attributes
        self.__cub_gen = cub_gen
        self.__video_info = copy(cub_gen.video_info)

        # Configure the original cuboid generator
        self.__batch_size = self.__cub_gen.batch_size
        self.__return_cub_as_label = self.__cub_gen.return_cub_as_label

        self.__cub_gen.batch_size = 1
        self.__cub_gen.return_cub_as_label = False

        # Save cumulative count of number of consecutive frames' cuboids
        #  retrievable per video
        self.__access_frames = np.zeros((len(self.__video_info)), dtype='int32')

        self.__video_info[0]['cum frames'] = self.__video_info[0]['frames']
        self.__access_frames[0] = max(1, self.__video_info[0]['frames'] - cub_gen.cub_frames)

        if self.__video_info[0]['frames'] % cub_gen.cub_frames != 0:
            # Add the needed frames to conform a video dividible by the number
            # of cuboids
            self.__video_info[0]['cum frames'] += (
                    cub_gen.cub_frames - self.__video_info[0]['frames'] % cub_gen.cub_frames)
            self.__access_frames[0] += (cub_gen.cub_frames - self.__video_info[0]['frames'] % cub_gen.cub_frames)

        for i in range(1, len(self.__video_info)):
            self.__video_info[i]['cum frames'] = (
                    self.__video_info[i - 1]['cum frames'] + self.__video_info[i]['frames'])

            self.__access_frames[i] = (
                    self.__access_frames[i - 1] + max(1, self.__video_info[i]['frames'] - cub_gen.cub_frames))

            if self.__video_info[i]['frames'] % cub_gen.cub_frames != 0:
                # Add the needed frames to conform a video dividible by the number
                # of cuboids
                self.__video_info[i]['cum frames'] += (
                        cub_gen.cub_frames - self.__video_info[i]['frames'] % cub_gen.cub_frames)
                self.__access_frames[i] += (cub_gen.cub_frames - self.__video_info[i]['frames'] % cub_gen.cub_frames)

        self.__frames = None
        self.__loaded_frame_range = [None, None]

    @property
    def return_cub_as_label(self):
        return self.__return_cub_as_label

    @property
    def batch_size(self):
        return self.__batch_size

    def __len__(self):
        return np.ceil(self.__access_frames[-1] / self.batch_size).astype(int)

    @property
    def num_cuboids(self):
        """Returns the real number of cuboids retrievable"""
        return self.__access_frames[-1]

    def __get_cuboid(self, idx: int):

        if idx < 0:
            idx %= self.__access_frames[-1]

        if idx >= self.__access_frames[-1]:
            raise IndexError('Index out of range')

        if self.__frames is None or not (self.__loaded_frame_range[0] <= idx < self.__loaded_frame_range[1]):
            # Look for the video containing the desired frames
            vid_idx = np.searchsorted(self.__access_frames, idx, side='right')

            self.__loaded_frame_range[0] = self.__access_frames[vid_idx - 1] if vid_idx > 0 else 0
            self.__loaded_frame_range[1] = self.__access_frames[vid_idx]

            """
            start = (self.__video_info[vid_idx-1]['frames'] +
                        self.__cub_gen.cub_frames -
                    self.__video_info[vid_idx-1]['frames'] %
                        self.__cub_gen.cub_frames) if vid_idx > 0 else 0
            """
            start = self.__video_info[vid_idx - 1]['cum frames'] // self.__cub_gen.cub_frames if vid_idx > 0 else 0
            stop = self.__video_info[vid_idx]['cum frames'] // self.__cub_gen.cub_frames

            """
            stop = np.ceil((start + self.__video_info[vid_idx]['frames'] +
                            self.__cub_gen.cub_frames -
                            self.__video_info[vid_idx]['frames'] %
                            self.__cub_gen.cub_frames )/8).astype(int)
            start = np.ceil(start / 8).astype(int)
            """

            self.__frames = self.__cub_gen[start: stop]
            self.__frames = self.__frames.reshape(self.__frames.shape[0] * self.__frames.shape[1],
                                                  *self.__frames.shape[2:])

        # The desired frames are already loaded on RAM so it can be returned
        # directly
        idx -= self.__loaded_frame_range[0]
        ret = self.__frames[idx: idx + self.__cub_gen.cub_frames]

        # The cuboid is not complete so it must be fill with repetitions
        # of last frame
        if ret.shape[0] != self.__cub_gen.cub_frames:
            append = np.repeat(ret[np.newaxis, -1], self.__cub_gen.cub_frames - ret.shape[0], axis=0)

            ret = np.concatenate((ret, append), axis=0)

        ret = np.expand_dims(ret, axis=0)

        return ret

    def __getitem__(self, idx):

        if self.__batch_size == 1:
            ret = self.__get_cuboid(idx)
        else:
            start = idx * self.__batch_size
            stop = min(idx * self.__batch_size + self.__batch_size, self.__access_frames[-1])

            # Get first cuboid and allocate for the remained cuboids
            f_cub = self.__get_cuboid(start)[0]

            ret = np.zeros((stop - start, *f_cub.shape), dtype=f_cub.dtype)
            ret[0] = f_cub

            # Retrieve the remain cuboids
            for i in range(stop - start):
                ret[i] = self.__get_cuboid(i + start)[0]

        return ret if not self.return_cub_as_label else (ret, ret)

    @property
    def cum_cuboids_per_video(self):
        return tuple(self.__access_frames)


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

    def __init__(self, cub_frames: int, *args: Union[List, Tuple, np.ndarray, CuboidsGenerator]):

        # Check input
        if not isinstance(cub_frames, int) or cub_frames <= 0:
            raise ValueError('"cub_frames" must be an integer greater than 0')

        if len(args) > 1 and any(not isinstance(x, np.ndarray) or len(x) != cub_frames for x in args):
            raise ValueError('Any cuboid provided not a valid numpy array or not have the number of frames specified')
        elif not (hasattr(args[0], '__getitem__') and hasattr(args[0], '__len__')):
            raise ValueError('cuboids collection passed must support an item getter and len operator')

        # Copy to private attributes
        if len(args) == 1:
            args = args[0]

        self._cuboids = args
        self.__cub_frames = cub_frames

    def __len__(self):

        """Returns the total number of frames to be retrievable
        """
        return len(self._cuboids) * self.__cub_frames

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
        return self._cuboids[idx // self.__cub_frames][idx % self.__cub_frames]
