# -*- coding: utf-8 -*-
###############################################################################
# Author: Nicolás Cubero Torres
# Description: Several utilities for the deployment and training of an
#            Incremental Spatio Temporal Learner model (ISTL). The ISTL model
#            is an unsupervised deep-learning approach for surveillance anomaly
#            detection that learns to reconstruct cuboids of video-frames
#            representing the normal behaviour with the lower loss, anomalies
#            are detected by the greater reconstruction error given by videos
#            which contains anomalies.
#
#            The ISTL model utilizes active learning with fuzzy aggregation,
#            to continuously update its knowledge of new anomalies and normality
#            that evolve over time.
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
from bisect import bisect_right

import numpy as np
from sklearn.metrics import roc_auc_score
from tensorflow import image as tf_image
from tensorflow import math as tf_math
from tensorflow import transpose as tf_transpose
from tensorflow.keras import Model, Sequential, Input
from tensorflow.keras.layers import (Conv2D, ConvLSTM2D, Conv2DTranspose,
                                     TimeDistributed, LayerNormalization,
                                     Lambda, Reshape)
from tensorflow.keras.layers import Conv3D, Conv3DTranspose

from utils import confusion_matrix, equal_error_rate


# from persistence1d.filter_noise import filter_noise

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
    istl.add(TimeDistributed(Conv2D(filters=128, kernel_size=(27, 27),
                                    strides=(4, 4), name='C1', padding='same', activation='tanh')))
    istl.add(LayerNormalization())
    """
        C2: Second Convolutional 2D layer.
        Kernel size: 13x13
        Filters: 64
        Strides: 2x2
    """
    istl.add(TimeDistributed(Conv2D(filters=64, kernel_size=(13, 13),
                                    strides=(2, 2), name='C2', padding='same', activation='tanh')))
    istl.add(LayerNormalization())
    """
        CL1: First Convolutional LSTM 2D layer
        Kernel size: 3x3
        Filters: 64
    """
    istl.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), name='CL1',
                        return_sequences=True, padding='same',
                        dropout=0.4, recurrent_dropout=0.3))
    # dropout=0.1,recurrent_dropout=0.05
    istl.add(LayerNormalization())
    """
        CL2: Second Convolutional LSTM 2D layer
        Kernel size: 3x3
        Filters: 32
    """
    istl.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), name='CL2',
                        return_sequences=True, padding='same',
                        dropout=0.3))
    istl.add(LayerNormalization())
    """
        DCL1: Third Convolutional LSTM 2D layer used for preparing deconvolution
        Kernel size: 3x3
        Filters: 64
    """
    istl.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), name='DCL1',
                        return_sequences=True, padding='same',
                        dropout=0.5))
    istl.add(LayerNormalization())
    """
        DC1: First Deconvolution 2D Layer
        Kernel size: 13x13
        Filters: 64
        Strides: 2x2
    """
    istl.add(TimeDistributed(Conv2DTranspose(filters=128, kernel_size=(13, 13),
                                             strides=(2, 2), name='DC1', padding='same', activation='tanh')))
    istl.add(LayerNormalization())
    """
        DC2: Second Deconvolution 2D Layer
        Kernel size: 27x27
        Filters: 128
        Strides: 4x4
    """
    istl.add(TimeDistributed(Conv2DTranspose(filters=1, kernel_size=(27, 27),
                                             strides=(4, 4), name='DC2', padding='same', activation='tanh')))

    return istl


def build_abnor_evant_STA():
    """
    Return the model used for abnormal event
    detection in videos using spatiotemporal autoencoder
    """
    model = Sequential()

    model.add(Conv3D(filters=128, kernel_size=(1, 11, 11), strides=(1, 4, 4,),
                     padding='valid', input_shape=(10, 227, 227, 1), activation='tanh'))

    model.add(Conv3D(filters=64, kernel_size=(1, 5, 5), strides=(1, 2, 2),
                     padding='valid', activation='tanh'))

    model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), strides=1, padding='same',
                         dropout=0.4, recurrent_dropout=0.3, return_sequences=True))

    model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), strides=1,
                         padding='same', dropout=0.3, return_sequences=True))

    model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), strides=1,
                         return_sequences=True, padding='same', dropout=0.5))

    model.add(Conv3DTranspose(filters=128, kernel_size=(1, 5, 5), strides=(1, 2, 2),
                              padding='valid', activation='tanh'))
    model.add(Conv3DTranspose(filters=1, kernel_size=(1, 11, 11), strides=(1, 4, 4),
                              padding='valid', activation='tanh'))

    return model


def root_sum_squared_error(inputs):
    # if K.ndim(y_true) > 2:
    #    return K.mean(K.sqrt(K.sum(K.square(y_true - y_pred),
    #                axis=K.arange(1, K.ndim(y_true)) )))
    # else:
    return tf_math.sqrt(tf_math.reduce_sum(tf_math.square(inputs[0] - inputs[1]), axis=(1, 2, 3, 4)))


class ScorerISTL:
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
            raise ValueError('"model" should be a Keras model containing a ' \
                             'pretrained ISTL model')

        if not isinstance(cub_frames, int) or cub_frames <= 0:
            raise ValueError('"cub_frames" must be an integer greater than 0')

        # Copy to the object atributes
        self.__model = model
        self.__cub_frames = cub_frames

        # Add extra layer to the model for the parallel computation of reconstrucion error
        self.__rec_error = Lambda(root_sum_squared_error)(
            [self.__model.layers[0].input, self.__model.layers[-1].output])
        self._rec_model = Model(inputs=self.__model.layers[0].input, outputs=self.__rec_error)

        # The minimum and maximum reconstruction error values commited by the
        # input model for the training cuboids used for normalize scores
        self.__min_score_cub = None
        self.__max_score_cub = None

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
            raise TypeError('"cuboids" must be a numpy array')

        # if not isinstance(scale_scores, bool):
        #    raise TypeError('"scales_scores" must be boolean')

        # Check if the scorer has been trained
        # if scale_scores and self.__min_score_cub is None:
        #    raise RuntimeError('Fitting to the training cuboids score '\
        #                        'is required first to return the scaled scores')

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

        # score = np.sqrt(np.sum((cuboid - self.__model.predict(cuboid))**2))
        score = self._rec_model.predict(cuboid)

        # if scale_scores:
        #    score = (score - self.__min_score_cub) / self.__max_score_cub

        return score

    def score_cuboids(self, cub_set: np.array or list or tuple,
                      scale_scores=True, norm_zero_one=False):

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
        if not hasattr(cub_set, '__getitem__') or not hasattr(cub_set, '__len__'):
            raise ValueError('Input cuboid\'s collection must have __getitem__ and __len__ methods')

        if not isinstance(scale_scores, (bool, tuple, list, np.ndarray)):
            raise TypeError('scale_scores must be bool, tuple, list or numpy array')

        if not isinstance(norm_zero_one, bool):
            raise TypeError('"norm_zero_one" must be bool')

        # Procedure
        """
        ret = np.zeros(len(cub_set), dtype='float64')

        for i in range(len(cub_set)):
            ret[i] = self.score_cuboid(cub_set[i])
        """
        ret = self._rec_model.predict(cub_set)
        ret = self._scale_scores(ret, scale_scores, norm_zero_one)

        return ret

    def _scale_scores(self, scores: np.array,
                      scale_scores=True, norm_zero_one=False):

        if isinstance(scale_scores, bool) and scale_scores:

            if self.__min_score_cub is None:
                min_value = scores.min()
                max_value = scores.max()
            else:
                min_value = self.__min_score_cub
                max_value = self.__max_score_cub

            scores = ((scores - min_value) / (max_value - min_value), scores)
        # ret = ((ret - min_value)/max_value, ret)

        elif isinstance(scale_scores, (tuple, list, np.ndarray)):

            scores_norm = np.zeros(scores.size, dtype=scores.dtype)

            # Perform normalization per video
            for i in range(len(scale_scores)):

                start = scale_scores[i - 1] if i > 0 else 0  # Starting video cuboid
                end = scale_scores[i]  # Ending video cuboid

                min_value = scores[start: end].min()
                max_value = scores[start: end].max()
                # min_value = ret[start: end].mean()
                # max_value = ret[start: end].std()

                if not norm_zero_one:
                    scores_norm[start: end] = ((scores[start: end] - min_value) / max_value)
                else:
                    scores_norm[start: end] = ((scores[start: end] - min_value) / (max_value - min_value))

            # Filter the more persistence optima
            # ret_norm[start: end] = filter_noise(ret_norm[start: end],
            #                        (ret_norm[start: end].max() -
            #                            ret_norm[start: end].min())*0.2)
            scores = (scores_norm, scores)

        return scores

    def fit(self, cub_set: np.array or list or tuple):

        """Fits the Scorer to the scores evaluated for the input cuboids
            collection so that the scoring can be scaled to the data learned
            on this fit
        """

        scores = self.score_cuboids(cub_set, False)
        self.__min_score_cub, self.__max_score_cub = scores.min(), scores.max()
        # self.__min_score_cub, self.__max_score_cub = scores.mean(), scores.std()

        return scores


class PredictorISTL(ScorerISTL):
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

        super(PredictorISTL, self).__init__(model, cub_frames)

        # Check input
        if (not isinstance(anom_thresh, (float, int)) or anom_thresh < 0 or
                anom_thresh > 1):
            raise ValueError('"anom_thresh" must be a float in [0, 1]')

        if not isinstance(temp_thresh, int) or temp_thresh < 0:
            raise ValueError('"temp_thresh" must be an integer greater or equal than 0')

        # Copy to the object atributes
        self.__anom_thresh = anom_thresh
        self.__temp_thresh = temp_thresh

    @property
    def anom_thresh(self):
        return self.__anom_thresh

    @property
    def temp_thresh(self):
        return self.__temp_thresh

    @anom_thresh.setter
    def anom_thresh(self, value: float):

        if not isinstance(value, (float, int)) or not 0 <= value <= 1:
            raise ValueError('"anom_thresh" must be a float in [0, 1]')
        self.__anom_thresh = value

    @temp_thresh.setter
    def temp_thresh(self, value: int):

        if not isinstance(value, int) or value < 0:
            raise ValueError('"temp_thresh" must be an integer greater or equal than 0')

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

        # return (self.score_cuboids(np.expand_dims(cuboid, axis=0), True)[0] >
        #                                                    self.__anom_thresh)
        return self.score_cuboid(cuboid)[0] > self.__anom_thresh

    def predict_cuboids(self, cub_set, return_scores=False,
                        cum_cuboids_per_video: list or tuple or np.ndarray = None,
                        norm_zero_one=False) -> np.ndarray:

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

                cum_cuboids_per_video : array-like
                    array, list or tuple containing the cumulative cuboids
                    of each video.

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
        scale = cum_cuboids_per_video if cum_cuboids_per_video is not None else True

        score, true_score = self.score_cuboids(cub_set, scale, norm_zero_one)
        preds = PredictorISTL._predict_from_scores(score, self.__anom_thresh,
                                                   self.__temp_thresh)

        return preds if not return_scores else (preds, score, true_score)

    @staticmethod
    def _predict_from_scores(score: np.ndarray, anom_thresh: float or int, temp_thresh: int):

        # Get cuboids whose reconstruction error overpass the anomalous threshold
        anom_cub = score >= anom_thresh

        preds = np.zeros(len(score), dtype='int8')

        # Count if there's temp_thresh consecutive anomalous cuboids
        cons_count = 0
        i_start = None
        for i in range(anom_cub.size):

            if not anom_cub[i]:
                if cons_count >= temp_thresh:
                    preds[i_start: i] = 1

                cons_count = 0
                i_start = None

            else:

                if i_start is None:
                    i_start = i

                cons_count += 1

        # Note the remainds cuboids
        if cons_count and cons_count >= temp_thresh:
            preds[i_start:] = 1

        return preds

    """
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
    """


class LocalizatorISTL(PredictorISTL):
    """Handler class for both temporal and spacial anomaly localization by a
        previous trained ISTL model throgh the original method proposed by [1].

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
            consider a segment as anomalous.

        subwind_size: array type containing two int values.
            Size of the sub-windows in which the original cuboids will be
            subdivided to analyse the anomalies' spatial location

    """

    def __init__(self, model: Model, cub_frames: int, anom_thresh: float, temp_thresh: int, subwind_size: tuple):
        super(LocalizatorISTL, self).__init__(model, cub_frames, anom_thresh, temp_thresh)
        # Private attributes
        self.subwind_size = subwind_size
        self._loc_model, self._base_loc_model = self.__build_localizator_model()

    @property
    def subwind_size(self):
        return self.__subwind_size

    @subwind_size.setter
    def subwind_size(self, subwind_size):

        if not (hasattr(subwind_size, '__getitem__') and hasattr(subwind_size, '__len__')):
            raise TypeError('"subwind_size" must be an array type object')

        if not (len(subwind_size) == 2 and
                all(isinstance(v, int) and v > 0 for v in subwind_size)):
            raise ValueError('"subwind_size" must be a two-int tuple greater than 0')

        self.__subwind_size = subwind_size

    def predict_cuboids(self, cub_set, return_scores=False,
                        cum_cuboids_per_video: list or tuple or np.ndarray = None,
                        norm_zero_one=False, only_tensors=True) -> tuple:

        """For each cuboid retrievable from a cuboid's collection (i.e.
            array-like object of cuboids or any genereator of cuboids), predicts
            wheter the cuboid is anomalous or not (i.e. its represents an
            anormal event or a normal event) and localizes the anomalies spatial
            location.

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

                cum_cuboids_per_video : array-like
                    array, list or tuple containing the cumulative cuboids
                    of each video.

                only_tensors: bool
                    Set True to make all the spatial analysis through Tensorial
                    operations which are executed on GPU (if Tensorflow is
                    configured to do so) which uses more memory of False to
                    let the CPU to perform the first preprocessing steps.

            Raise
            -----

            ValueError: cub_set is not indexable or its length cannot be known
            ValueError: Any cuboid is not a valid numpy ndarray

            Return:
                - 8-bit int numpy array vector with the prediction
                for each collection's cuboid if return_scores is False
                or a tuple with the prediction vector and the 64-bit float
                numpy array vector containing the reconstruction error
                associated to each cuboid.

                - Dict specifying for each anomalous cuboid the upper left
                corners of anomalous subwindows detected
        """

        # Check input
        if not isinstance(only_tensors, bool):
            raise TypeError('only_tensors must be boolean')

        # Get prediction and scores if returned
        ret = super(EvaluatorISTL, self).predict_cuboids(
            cub_set,
            return_scores,
            cum_cuboids_per_video,
            norm_zero_one)

        if not isinstance(ret, tuple):
            ret = (ret,)

        # Perform spatial-analysis on anomalous cuboids
        preds = ret[0]

        det = self.spatial_loc_anomalies(cub_set, preds, only_tensors)

        ret = ret + (det,)
        return ret

    def __build_localizator_model(self):

        """
        original_shape = cuboid.shape[1:]
        split_cub_shape = (cuboid.shape[1],
                            cuboid.shape[2]//self.__subwind_size[0],
                            self.__subwind_size[0],
                            cuboid.shape[3]//self.__subwind_size[1],
                            self.__subwind_size[1], cuboid.shape[4])

        split_cub = cuboid.reshape(split_cub_shape)

        # Shift the cumuled axis to the first axises
        split_cub = np.rollaxis(split_cub, 1, 0)
        split_cub = np.rollaxis(split_cub, 3, 1)

        split_cub = split_cub.reshape((-1,) + split_cub.shape[2:])
        """

        cub_length, width, height, channels = self._rec_model.input.shape[1:]
        split_cub_shape = (cub_length,
                           width // self.__subwind_size[0],
                           self.__subwind_size[0],
                           height // self.__subwind_size[1],
                           self.__subwind_size[1], channels)

        # Make base model
        base_input_layer = Input(shape=(cub_length, self.__subwind_size[0], self.__subwind_size[1], channels))
        base_resize_layer = TimeDistributed(Lambda(lambda x: tf_image.resize(x, (width, height))))(base_input_layer)
        base_rec_model = self._rec_model(base_resize_layer)

        base_model = Model(inputs=base_input_layer, outputs=base_rec_model)

        # Add resizing layers at first of to the rec model to resize the tensors
        # to the input accepted by the model
        input_layer = Input(shape=(cub_length, width, height, channels))
        # Split cuboid in subcuboids
        split_layer = Reshape(split_cub_shape)(input_layer)
        permut_dim_layer = Lambda(lambda x: tf_transpose(x, [0, 2, 4, 1, 3, 5, 6]))(split_layer)
        acum_dim_layer = Reshape((split_cub_shape[2] * split_cub_shape[4], cub_length, self.__subwind_size[0],
                                  self.__subwind_size[1], channels))(permut_dim_layer)
        # Resize each subcuboid at full cuboid size
        rec_model = TimeDistributed(base_model)(acum_dim_layer)
        # Predict each subcuboid
        # rec_model = self._rec_model(rec_model)

        model = Model(inputs=input_layer, outputs=rec_model)
        return model, base_model

        """
        cub_length, width, height, channels = self._rec_model.input.shape[1:]

        # Add resizing layers at first of to the rec model to resize the tensors
        # to the input accepted by the model
        input_layer = Input(shape=(cub_length, self.__subwind_size[0], 
                                    self.__subwind_size[1], channels))
        resize_layer = TimeDistributed(Lambda(lambda x: tf_image.resize(x, (width, height))) )(input_layer)
        rec_model = self._rec_model(resize_layer)

        model = Model(inputs=input_layer, outputs=rec_model)
        return model
        """

    def spatial_loc_anomalies(self, cub_set, preds=None, only_tensors=True):

        """
            Performs spatial location of a set of predicted cuboids through
            sub-windows scannation by the given sub-windows size.

            Parameters
            ----------

            cub_set: indexable and length-known collection of cuboids.
            (array, list or tuple of cuboids, generator of cuboids)
            Array, list, tuple or any generator of cuboids to be scored

            preds: 8-bits int Numpy array
            Binary array of predictions given for each cuboid of
            the cuboid set.

            subwind_size: 2-int tuple
            Tuple containing the width and the height of the desired subwindows.

            only_tensors: bool
                Set True to make all the spatial analysis through Tensorial
                operations which are executed on GPU (if Tensorflow is
                configured to do so) which uses more memory of False to
                let the CPU to perform the first preprocessing steps.

            Return
            ------
            Dict containing a list of the indexes of the anomalous sub-windows
                for each anomalous cuboid index
        """

        # Check input
        if not hasattr(cub_set, '__getitem__') or not hasattr(cub_set, '__len__'):
            raise TypeError('Input cuboid\'s collection must have ' \
                            '__getitem__ and __len__ methods')

        if preds is not None and (not hasattr(preds, '__getitem__') or not hasattr(preds, '__len__')):
            raise TypeError('Prediction array must be an array type containing' \
                            'a label for each cuboid')

        if preds is not None and len(cub_set) != len(preds):
            raise ValueError('The cuboid set and the pediction array must be' \
                             ' of the same length')

        if not isinstance(only_tensors, bool):
            raise TypeError('only_tensors must be boolean')
        """
        if not hasattr(subwind_size, ('__getitem__', '__len__')):
            raise TypeError('"subwind_size" must be an array type object')

        if not (len(subwind_size) != 2 and
                all(isinstance(v, int) and v > 0 for v in subwind_size)):
            raise ValueError('"subwind_size" must be a two-int tuple greater '\
                                'than 0')
        """

        pos_preds = np.where(preds == 1)[0] if preds is not None else np.arange(len(cub_set))

        rows = cub_set[0].shape[2]
        cols = cub_set[0].shape[3]

        idxs = np.array([[i, j] for i in range(0, rows, self.__subwind_size[0])
                         for j in range(0, cols, self.__subwind_size[1])])

        det = {}

        # Scan all the anomalous cuboids returning the location
        # of all anomalous sub-windows
        for i in pos_preds:
            ret = self.__cuboid_scannation(cub_set[i], idxs, only_tensors)

            if ret is not None:
                det[i] = ret

        return det

    def __cuboid_scannation(self, cuboid: np.ndarray, idxs: np.ndarray, only_tensors: bool):

        """
        # Remove the extra dimension
        if cuboid.shape[0] == 1:
            cuboid = cuboid.reshape(*cuboid.shape[1:])

        dst_shape = cuboid.shape[1:-1]
        areas = []

        for i, j in idxs:
            # Get the sub-window
            sli = cuboid[:, i: max(i + self.__subwind_size[0],
                                        cuboid.shape[1]),
                            j: max(j + self.__subwind_size[1],
                                        cuboid.shape[2])]

            # Resize the sliced cuboid to be scored through the model
            amp_cub = np.expand_dims(
                        np.array([resize(fr, dsize=dst_shape) for fr in sli]),
                        axis=(0,-1)) #np.apply_along_axis(resize, 0, cuboid, dsize=dst_shape) #LocalizatorISTL.__vect_resize(sli, dst_shape)

            if self.predict_cuboid(amp_cub):
                areas.append((i, j))

        return areas

        original_shape = cuboid.shape[1:]
        split_cub_shape = (cuboid.shape[1],
                            cuboid.shape[2]//self.__subwind_size[0],
                            self.__subwind_size[0],
                            cuboid.shape[3]//self.__subwind_size[1],
                            self.__subwind_size[1], cuboid.shape[4])

        split_cub = cuboid.reshape(split_cub_shape)

        # Shift the cumuled axis to the first axises
        split_cub = np.rollaxis(split_cub, 1, 0)
        split_cub = np.rollaxis(split_cub, 3, 1)

        split_cub = split_cub.reshape((-1,) + split_cub.shape[2:])
        """

        # Score each split by the localizator model
        if only_tensors:
            pred = self._loc_model.predict(cuboid)  # split_cub)
        else:
            original_shape = cuboid.shape[1:]
            split_cub_shape = (cuboid.shape[1],
                               cuboid.shape[2] // self.__subwind_size[0],
                               self.__subwind_size[0],
                               cuboid.shape[3] // self.__subwind_size[1],
                               self.__subwind_size[1], cuboid.shape[4])

            split_cub = cuboid.reshape(split_cub_shape)

            # Shift the cumuled axis to the first axises
            split_cub = np.rollaxis(split_cub, 1, 0)
            split_cub = np.rollaxis(split_cub, 3, 1)

            split_cub = split_cub.reshape((-1,) + split_cub.shape[2:])

            pred = self._base_loc_model.predict(split_cub)

        pred = self._scale_scores(pred, True)[0]
        pred = pred > self.anom_thresh

        return idxs[pred] if pred.any() else None


# __vect_resize = np.vectorize(resize, excluded={'dsize', 'fx', 'fy', 'interpolation'})

class EvaluatorISTL(PredictorISTL):
    """Handler class for the performance evaluation of a ISTL model prediction.
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
                 temp_thresh: int, max_cuboids: int = None):
        super(EvaluatorISTL, self).__init__(model, cub_frames, anom_thresh,
                                            temp_thresh)

        # Private attributes
        self.__fp_cuboids = []  # List of false positive stored cuboid
        self.__fp_rec_errors = []  # Reconstruction error of stored cuboid

        self.__max_cuboids = max_cuboids

    @property
    def fp_cuboids(self):
        if len(self.__fp_cuboids) > 1:
            return np.array(self.__fp_cuboids)

        if len(self.__fp_cuboids) == 1:
            return np.expand_dims(self.__fp_cuboids, axis=0)

        return None

    # return (np.array(self.__fp_cuboids) if len(self.__fp_cuboids) > 1 else
    #                            np.expand_dims(self.__fp_cuboids, axis=0))

    def __len__(self):
        return len(self.__fp_cuboids)

    ## Methods ##
    def evaluate_cuboids(self, cuboids: np.ndarray, labels: list or np.ndarray,
                         cum_cuboids_per_video: list or tuple or np.ndarray = None,
                         norm_zero_one=False, return_rec_errors=True):

        """Evaluates the prediction of the trained ISTL model given the anomaly
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
            raise ValueError('The passed cuboids are not a valid collection of cuboids')

        if not hasattr(labels, '__getitem__') or not hasattr(labels, '__len__'):
            raise ValueError('The passed labels is not a valid collection of integers')

        if len(cuboids) != len(labels):
            raise ValueError('There must be a label for each cuboid')

        # Procedure
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)

        # Predict all cuboids
        pred, scores, true_scores = self.predict_cuboids(cub_set=cuboids,
                                                         return_scores=True,
                                                         cum_cuboids_per_video=cum_cuboids_per_video,
                                                         norm_zero_one=norm_zero_one)

        # Compute reconstruction errors for each class
        rec_errors = {}

        if return_rec_errors:
            rec_errors['dist_abnormal_class'] = [float(val) for val in scores[labels == 1]]
            rec_errors['dist_normal_class'] = [float(val) for val in scores[labels == 0]]

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

        ret = EvaluatorISTL._compute_perf_metrics(labels, pred, scores,
                                                  true_scores)

        # Append the reconstruction error per class
        ret['reconstruction_error_norm'].update(rec_errors)
        return ret

    def _compute_perf_metrics(labels, pred, scores, true_scores=None):

        # Compute performance metrics
        cm = confusion_matrix(labels, pred)

        """
        # Normalize scores to compute AUC and EER
        scores_min = scores.min()
        scores_max = scores.max()
        scores_norm = (scores - scores_min) / (scores_max - scores_min)
        """

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

        ret = {
            'accuracy': float((cm[1, 1] + cm[0, 0]) / len(scores)),
            'precision': float(cm[1, 1] / (cm[1, 1] + cm[0, 1])),
            'recall': float(cm[1, 1] / (cm[1, 1] + cm[1, 0])),
            'specificity': float(cm[0, 0] / (cm[0, 0] + cm[0, 1])),
            'AUC': float(auc),
            'EER': float(eer)
        }

        try:
            ret['f1 score'] = ((2 * ret['precision'] * ret['recall']) /
                               (ret['precision'] + ret['recall']))
        except Exception as e:
            ret['f1 score'] = float(np.NaN)
            warnings.warn(str(e))

        ret['TPRxTNR'] = ret['recall'] * ret['specificity']

        ret['confusion matrix'] = {
            'TP': int(cm[1, 1]),
            'TN': int(cm[0, 0]),
            'FP': int(cm[0, 1]),
            'FN': int(cm[1, 0])
        }

        ret['reconstruction_error_norm'] = {
            'mean': float(scores.mean()),
            'std': float(scores.std()),
            'min': float(scores.min()),
            'max': float(scores.max())
        }

        ret['class_reconstruction_error_norm'] = {}

        try:
            ret['class_reconstruction_error_norm']['Normal'] = {
                'mean': float(scores[labels == 0].mean()),
                'std': float(scores[labels == 0].std()),
                'min': float(scores[labels == 0].min()),
                'max': float(scores[labels == 0].max())
            }
        except Exception as e:
            ret['class_reconstruction_error_norm']['Normal'] = {
                'mean': float(np.NaN),
                'std': float(np.NaN),
                'min': float(np.NaN),
                'max': float(np.NaN)
            }

            warnings.warn('Couldn\'t compute normalized reconstruction error ' \
                          'summary for Normal Class: ' + str(e))

        try:
            ret['class_reconstruction_error_norm']['Abnormal'] = {
                'mean': float(scores[labels == 1].mean()),
                'std': float(scores[labels == 1].std()),
                'min': float(scores[labels == 1].min()),
                'max': float(scores[labels == 1].max())
            }
        except Exception as e:
            ret['class_reconstruction_error_norm']['Abnormal'] = {
                'mean': float(np.NaN),
                'std': float(np.NaN),
                'min': float(np.NaN),
                'max': float(np.NaN)
            }

            warnings.warn('Couldn\'t compute normalized reconstruction error ' \
                          'summary for Abnormal Class: ' + str(e))

        if true_scores is not None:
            ret['reconstruction_error'] = {
                'mean': float(true_scores.mean()),
                'std': float(true_scores.std()),
                'min': float(true_scores.min()),
                'max': float(true_scores.max())
            }

            ret['class_reconstruction_error'] = {}

            try:
                ret['class_reconstruction_error']['Normal'] = {
                    'mean': float(true_scores[labels == 0].mean()),
                    'std': float(true_scores[labels == 0].std()),
                    'min': float(true_scores[labels == 0].min()),
                    'max': float(true_scores[labels == 0].max())
                }
            except Exception as e:
                ret['class_reconstruction_error']['Normal'] = {
                    'mean': float(np.NaN),
                    'std': float(np.NaN),
                    'min': float(np.NaN),
                    'max': float(np.NaN)
                }
                warnings.warn('Couldn\'t compute reconstruction error ' \
                              'summary for Normal Class: ' + str(e))

            try:
                ret['class_reconstruction_error']['Abnormal'] = {
                    'mean': float(true_scores[labels == 1].mean()),
                    'std': float(true_scores[labels == 1].std()),
                    'min': float(true_scores[labels == 1].min()),
                    'max': float(true_scores[labels == 1].max())
                }
            except Exception as e:
                ret['class_reconstruction_error']['Abnormal'] = {
                    'mean': float(np.NaN),
                    'std': float(np.NaN),
                    'min': float(np.NaN),
                    'max': float(np.NaN)
                }

                warnings.warn('Couldn\'t compute reconstruction error ' \
                              'summary for Abnormal Class: ' + str(e))

        return ret

    def evaluate_cuboids_range_params(self, cuboids: np.ndarray,
                                      labels: list or np.ndarray,
                                      anom_thresh_range: list or tuple or np.ndarray,
                                      temp_thresh_range: list or tuple or np.ndarray,
                                      cum_cuboids_per_video: list or tuple or np.ndarray = None,
                                      norm_zero_one=False):

        """Evaluates the prediction of the trained ISTL model given each combined
            pair of anomaly threshold and temporal threshold values from the
            range of values specified foe each one.

            Parameters
            ----------

            cuboids : collection (list, tuple, numpy array) of cuboids
                Collection of cuboids to be classified

            labels : collection of integers
                Labels indicating 1 for anomaly or 0 for normal
        """

        # Check input
        if not hasattr(cuboids, '__getitem__') or not hasattr(cuboids, '__len__'):
            raise ValueError('The passed cuboids are not a valid collection' \
                             ' of cuboids')

        if not hasattr(labels, '__getitem__') or not hasattr(labels, '__len__'):
            raise ValueError('The passed labels is not a valid collection of integers')

        if len(cuboids) != len(labels):
            raise ValueError('There must be a label for each cuboid')

        if (not hasattr(anom_thresh_range, '__getitem__') or
                not hasattr(anom_thresh_range, '__len__')):
            raise TypeError('"anom_thresh_range" must be an indexable ' \
                            'collection of float/int values')

        if any((x < 0 or x > 1) for x in anom_thresh_range):  # not isinstance(x, (float, int)) or
            raise ValueError('Any anom threshold provided not a float or int ' \
                             ' in [0, 1]')

        if (not hasattr(temp_thresh_range, '__getitem__') or
                not hasattr(temp_thresh_range, '__len__')):
            raise TypeError('"temp_thresh_range" must be an indexable ' \
                            'collection of int values')

        if any(x < 0 for x in temp_thresh_range):  # not isinstance(x, int) or
            raise ValueError('Any temp thresh provided not an integer greater ' \
                             ' or equal than 0')

        # Procedure
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)

        # Predict all cuboids
        scale = cum_cuboids_per_video if cum_cuboids_per_video is not None else True

        scores, true_scores = self.score_cuboids(cuboids, scale, norm_zero_one)
        meas = {
            'reconstruction_error_norm': {
                'mean': float(scores.mean()),
                'std': float(scores.std()),
                'min': float(scores.min()),
                'max': float(scores.max()),
                'dist_abnormal_class': [float(val) for val in scores[labels == 1]],
                'dist_normal_class': [float(val) for val in scores[labels == 0]]
            },
            'reconstruction_error': {
                'mean': float(true_scores.mean()),
                'std': float(true_scores.std()),
                'min': float(true_scores.min()),
                'max': float(true_scores.max())
            },
            'results': []
        }

        # For each combined pair of anom thresh and temp thresh compute metrics
        for at in anom_thresh_range:
            for tt in temp_thresh_range:
                meas['results'].append({'anom_thresh': float(at),
                                        'temp_thresh': int(tt)})

                preds = PredictorISTL._predict_from_scores(scores, at, tt)
                meas['results'][-1].update(
                    EvaluatorISTL._compute_perf_metrics(labels, preds,
                                                        scores, true_scores))
        return meas

    def clear(self):

        """Clears all the false positive cuboids stored
        """

        self.__fp_cuboids = []
        self.__fp_rec_errors = []
