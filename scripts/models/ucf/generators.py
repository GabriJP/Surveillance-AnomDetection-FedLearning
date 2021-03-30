# -*- coding: utf-8 -*-
###############################################################################
# WARNING: THIS TOOL WAS NOT FINALLY INCLUDED IN THE PROJECT SO THE
#   IMPLEMENTATION IS NOT COMPLETE NOR TESTED
#
# Author: Nicolás Cubero Torres
# Description: Several utilities for the retrieval of video features data from
#                datasets as mini-batches composed by a random choices of
#                both class' samples
###############################################################################

# Imported modules
import os
import random
import warnings
from copy import copy

import numpy as np
from tensorflow.keras.utils import Sequence


class FeatureGenerator(Sequence):
    """Utility for the retrieval of mini-batches of video feature samples of
        both anomaly and normal classes at equal proportion given several source
        directories
    """

    def __init__(self, normal_src_dir: list or tuple or str,
                 anormal_src_dir: list or str, feature_size: int,
                 batch_size=32, shuffle=False,
                 seed: int = None, max_videos: int = 100):

        # Check input
        if not isinstance(normal_src_dir, (list, tuple, str)):
            raise TypeError('"normal_src_dir" must be a str or a collection of str')

        if isinstance(normal_src_dir, (list, tuple)) and any(not isinstance(x, str) for x in normal_src_dir):
            raise TypeError('Any source dir in "normal_src_dir" not a valid str')

        if not isinstance(anormal_src_dir, (list, tuple, str)):
            raise TypeError('"anormal_src_dir" must be a str or a collection of str')

        if isinstance(anormal_src_dir, (list, tuple)) and any(not isinstance(x, str) for x in anormal_src_dir):
            raise TypeError('Any source dir in "anormal_src_dir" not a valid str')

        if not isinstance(feature_size, int):
            raise TypeError('The feature size must be integer')

        if feature_size <= 0:
            raise ValueError('The feature size must be greater than 0')

        if not isinstance(batch_size, int):
            raise TypeError('The batch size must be integer')

        if batch_size <= 0:
            raise ValueError('Batch size must be greater than 0')

        if not isinstance(shuffle, bool):
            raise TypeError('"shuffle" must be boolean')

        if seed is not None and not isinstance(seed, int):
            raise ValueError('"seed" must be None or integer')

        if not isinstance(max_videos, int):
            raise TypeError('"max_videos" not a valid integer')

        if max_videos <= 0:
            raise ValueError('"max_videos" must be greater than 0')

        if batch_size and max_videos < batch_size:
            raise ValueError('The batch size cannot be greater than the max videos')

        # Define attributes
        if isinstance(normal_src_dir, (tuple, list)):
            self.__normal_src_dir = normal_src_dir
        else:
            self.__normal_src_dir = (normal_src_dir,)

        if isinstance(anormal_src_dir, (tuple, list)):
            self.__anormal_src_dir = anormal_src_dir
        else:
            self.__anormal_src_dir = (anormal_src_dir,)

        self.__feat_size = feature_size
        self.__batch_size = batch_size
        self.__shuffle = shuffle
        self.__seed = seed
        self.__max_videos = max_videos

        # Maximum number of batches to load from disk
        self.__max_batch = self.max_videos // self.__batch_size

        # Scan the specified directories looking for all the videos
        self.__normal_vid_info = FeatureGenerator.__scan_source_dir(
            self.__normal_src_dir)
        self.__anormal_vid_info = FeatureGenerator.__scan_source_dir(
            self.__anormal_src_dir)

        if not self.__normal_vid_info:
            raise RuntimeError('No normal video found on the specified directories')

        if not self.__anormal_vid_info:
            raise RuntimeError('No anormal video found on the specified directories')

        # Auxiliar vectors to get the video's feature in different order
        self.__acc_normal_vid_info = copy(self.__normal_vid_info)
        self.__acc_anormal_vid_info = copy(self.__anormal_vid_info)

        # To load the videos features
        self.__normal_vid = None
        self.__anormal_vid = None

        # Store the video's loaded range
        self.__loaded_vid_range = [None, None]

    @staticmethod
    def __scan_source_dir(dirs: tuple):

        video_files = []

        for dir in dirs:

            if not os.path.isdir(dir):
                warnings.warn('"{}" not a valid directory and will be omitted')
                continue

            # Get the path of listed file
            video_files.extend([dir + '/' + fn for fn in sorted(
                os.listdir(dir)) if fn.endswith('.txt')])

        return video_files

    @staticmethod
    def __load_vid_feat(fpath: str, feat_size: int) -> np.ndarray:

        """Loads a precomputed video's features
        """

        with open(fpath, 'r') as f:
            words = f.read().split()
            feat = np.array([words[i * feat_size: i * feat_size + feat_size]
                             for i in range(len(words) // feat_size)], dtype='float32')

        return feat

    def __len__(self) -> int:

        """Returns the number of batches to be retrievable
        """

        return min(len(self.__normal_vid_info),
                   len(self.__anormal_vid_info)) // self.__batch_size

    def __getitem__(self, idx) -> np.ndarray:

        """ Retrieves the features batch located at index conformed by video's
            features of both anomaly and normal situations
        """

        vid_idx = self.__batch_size * idx

        if vid_idx < 0 or vid_idx >= len(self):
            raise IndexError('index out of range')

        # The desired video's features are not loaded so it must be loaded
        if self.__normal_vid is None or (self.__loaded_vid_range[0] <= vid_idx < self.__loaded_vid_range[1]):
            self.__loaded_vid_range[0] = vid_idx

            # Load half of max_videos of each class
            self.__loaded_vid_range[1] = min(vid_idx +
                                             self.__max_batch * self.__batch_size,
                                             len(self.__normal_vid_info),
                                             len(self.__anormal_vid_info)) // 2

            self.__normal_vid = np.concatenate([
                FeatureGenerator.__load_vid_feat(self.__acc_normal_vid_info[i],
                                                 self.__feat_size)
                for i in range(self.__loaded_vid_range[0], self.__loaded_vid_range[1])
            ])

            self.__anormal_vid = np.concatenate([
                FeatureGenerator.__load_vid_feat(self.__acc_anormal_vid_info[i],
                                                 self.__feat_size)
                for i in range(self.__loaded_vid_range[0], self.__loaded_vid_range[1])
            ])
            # The desired video's feature batch is actually stored on RAM and
            #    can be returned directly
            start = vid_idx - self.__loaded_vid_range[0]
            end = start + self.__batch_size // 2

            # The batch is conformed by the half of video's features of each class
            batch = np.concatenate((self.__anormal_vid[start:end],
                                    self.__normal_vid[start:end]))

            labels = np.zeros(len(batch), dtype='uint8')
            labels[len(labels) // 2:] = 1

        return batch, labels

    def shuffle(self, shuf=False, seed=None):
        """Shuffle randomly the video set or undo the shufflering making the
            video set to recover its original order

            Parameters
            ----------

            shuf : bool
                True for shuffle the videos set or False to make the videos set
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

            random.shuffle(self.__acc_normal_vid_info)
            random.shuffle(self.__acc_anormal_vid_info)

            # Recover the original random state
            if seed is not None:
                random.setstate(or_rand_state)

        else:
            self.__acc_normal_vid_info = copy(self.__normal_vid_info)
            self.__acc_anormal_vid_info = copy(self.__anormal_vid_info)
