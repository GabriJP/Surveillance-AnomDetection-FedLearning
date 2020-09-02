# -*- coding: utf-8 -*-
###############################################################################
# Author: Nicolás Cubero Torres
# Description: Several utilities implementation for training and testing the
#               model proposed for the detection of real world anomalies on
#               video surveillance.
#
#               The model proposed is a supervised classification deep neural
#               model able to detect and temporal-locate the anomaly events on
#               surveillance video.
#
# References:
#
# Sultani, Waqas, Chen Chen, and Mubarak Shah. "Real-world anomaly detection
#  in surveillance videos." Proceedings of the IEEE Conference on Computer
#  Vision and Pattern Recognition. 2018. arXiv:1801.04264
###############################################################################

# Imported modules
from tensorflow.keras import Model, Sequential, Input
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.losses import LossFunctionWrapper
from keras.regularizers import l2

def build_UCF_model():

    """Builder function for creating an empty UCF-Crime deep neural model
        which receives a 4096 length 1D-feature vector obtained from a ConvNet3D
        convolutional model applied over not overlaying video segments.
    """

    model = Sequential()

    """
        FC1: First fully-connected layer.
        Input: 4096 1D-feature vector.
        Units: 512
        Regularization: -> 10^-3
        Activation: RELU
        Dropout: 60 %
    """
    model.add(Dense(512, input_dim=4096, init='glorot_normal',
                    W_regularizer=l2(0.001), activation='relu', name='fc1'))
    model.add(Dropout(0.6))

    """
        FC2: Second fully-connected layer.
        Input: 512 length vector.
        Units: 32
        Regularization: L2 -> 10^-3
        Activation: Linear
        Dropout: 60 %
    """
    model.add(Dense(32, init='glorot_normal', W_regularizer=l2(0.001),
                    name='fc2'))
    model.add(Dropout(0.6))

    """
        FC1: Output layer.
        Input: 32-length vector.
        Units: 1
        Regularization: L2 -> 10^-3
        Activation: Sigmoid
    """
    model.add(Dense(1, init='glorot_normal', W_regularizer=l2(0.001),
                    activation='sigmoid', name='fc3'))

    return model

class UCFHingeLoss(LossFunctionWrapper):

    """Implementation of the measure loss used for the UCF-Crime model

        Parameters
        ----------

        reduction

        vid_seg: int
            Number of segments conforming a video

        lamb1 : float
            Temporal smoothness term's regularization parameter

        lamb2 : float
            Sparsity term's regularization parameter.
    """

    def __init__(self, reduction='auto', vid_seg=32,
                    lamb1=0.00008, lamb2=0.00008):

        super(UCFHingeLoss, self).__init__(objective_function, reduction,
                                            'UCF-crime_hinge_loss',
                                            vid_seg, lamb1, lamb2)

        # Check input
        if not isinstance(vid_seg, int):
            raise TypeError('"vid_seg" must be integer')

        if vid_seg <= 0:
            raise ValueError('"vid_seg" must be greater than 0')

        if not isinstance(lamb1, (float, int)):
            raise TypeError('"lamb1" must be float or integer value')

        if lamb1 < 0 or lamb1 > 1:
            raise ValueError('"lamb1" must be in [0, 1]')

        if not isinstance(lamb2, (float, int)):
            raise TypeError('"lamb2" must be float or integer value')

        if lamb2 < 0 or lamb2 > 1:
            raise ValueError('"lamb2" must be in [0, 1]')

def objective_function(y_true, y_pred, vid_seg=32,
                                            lamb1=0.00008, lamb2=0.00008):

    # Prepare true labels and predictions
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)

    y_true = K.reshape(y_true, (-1, vid_seg))
    y_pred = K.reshape(y_pred, (-1, vid_seg))

    # Know the number of videos
    n_vids = K.get_value(y_true.get_shape()[0])
    n_vids_per_class = n_vids // 2

    # Compute max scores and sum scores for each video bag
    #true_bag_max_score = K.max(y_true, axis=1)
    true_bag_sum_score = K.sum(y_true, axis=1)
    pred_bag_max_score = K.max(y_pred, axis=1)
    pred_bag_sum_score = K.sum(y_pred, axis=1)

    # Temporal smooth term
    temp_sm_term = K.sum(K.square(y_pred[:, :-1] - y_pred[:, 1:]))

    # Separate normal from abnormal videos
    anom_vid_index, normal_vid_index = (K.equal(true_bag_sum_score, 0),
                                        K.equal(true_bag_sum_score, 32))

    z = K.max(1 - pred_bag_max_score[anom_vid_index] +
                pred_bag_max_score[normal_vid_index], 0) +
        lamb1 * temp_sm_term + lamb2 * pred_bag_sum_score

def objective_function2(y_true, y_pred, lamb1=0.00008, lamb2=0.00008):

    #'Custom Objective function'

    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)

    n_seg = 32  # Because we have 32 segments per video.
    nvid = K.get_value(y_true.get_shape()[0])
    n_exp = nvid / 2
    Num_d=32*nvid


    sub_max = K.ones_like(y_pred) # sub_max represents the highest scoring instants in bags (videos).
    sub_sum_labels = K.ones_like(y_true) # It is used to sum the labels in order to distinguish between normal and abnormal videos.
    sub_sum_l1 = K.ones_like(y_true)  # For holding the concatenation of summation of scores in the bag.
    sub_l2 = K.ones_like(y_true) # For holding the concatenation of L2 of score in the bag.

    for i in range(nvid):

        # For Labels
        vid_seg = y_true[i * n_seg: i * n_seg + n_seg]
        # Esto coloca la suma de la puntuaciones de los segmentos del vídeo i en sub_sum_labels
        sub_sum_labels = K.concatenate([sub_sum_labels, K.stack(K.sum(vid_seg))])  # Just to keep track of abnormal and normal vidoes

        # For Features scores
        Feat_Score = y_pred[i * n_seg: i * n_seg + n_seg]
        # El primero coloca la puntuación del cuboide más anómalo en la posición
        # i de sub_max mientras que el segundo coloca la suma de las puntuaciones
        # de los segmentos en sub_sum_l1
        sub_max = K.concatenate([sub_max, K.stack(K.max(Feat_Score))])         # Keep the maximum score of scores of all instances in a Bag (video)
        sub_sum_l1 = K.concatenate([sub_sum_l1, K.stack(K.sum(Feat_Score))])   # Keep the sum of scores of all instances in a Bag (video)

        # Compute the temporal smoothness term
        z1 = T.ones_like(Feat_Score) # length = n_seg
        z2 = T.concatenate([z1, Feat_Score]) # length = 2*n_seg
        z3 = T.concatenate([Feat_Score, z1]) # length = 2*n_seg
        z_22 = z2[31:] # Esto sacaría la segunda parte (Feat_Score con un 1 delante) de z2
        z_44 = z3[:33] # Esto sacaría la primera partr (Feat_Score con un 1 detrás) de z3
        z = z_22 - z_44 # Aquí se estaría estando a cada valor de Feat_Score el valor que tiene en la posición i+1
        z = z[1:32]
        z = T.sum(T.sqr(z))

        # Save the temporal smoothness term on the i position of sub_l2
        sub_l2 = T.concatenate([sub_l2, T.stack(z)])


    # sub_max[Num_d:] means include all elements after Num_d.
    # AllLabels =[2 , 4, 3 ,9 ,6 ,12,7 ,18 ,9 ,14]
    # z=x[4:]
    #[  6.  12.   7.  18.   9.  14.]

    sub_score = sub_max[Num_d:]  # We need this step since we have used T.ones_like
    F_labels = sub_sum_labels[Num_d:] # We need this step since we have used T.ones_like
    #  F_labels contains integer 32 for normal video and 0 for abnormal videos. This because of labeling done at the end of "load_dataset_Train_batch"



    # AllLabels =[2 , 4, 3 ,9 ,6 ,12,7 ,18 ,9 ,14]
    # z=x[:4]
    # [ 2 4 3 9]... This shows 0 to 3 elements

    sub_sum_l1 = sub_sum_l1[Num_d:] # We need this step since we have used T.ones_like
    sub_sum_l1 = sub_sum_l1[:n_exp]
    sub_l2 = sub_l2[Num_d:]         # We need this step since we have used T.ones_like
    sub_l2 = sub_l2[:n_exp]

    # F_labels contiene la suma de las puntuaciones reales anormales
    # sub_score la puntuación predicha máxima de cada vídeo
    # sub_sum_l1 la suma de las puntuaciones predichas de los patrones anormales
    # sub_l2 el término de suavizado temporal sobre los patrones anormales

    # Se coge un vídeo normal con la máxima puntuación de anomalía
    indx_nor = K.equal(F_labels, 32).nonzero()[0]  # Index of normal videos: Since we labeled 1 for each of 32 segments of normal videos F_labels=32 for normal video
    # Se coge un vídeo anormal con la mínima puntuación de anomalía
    indx_abn = K.equal(F_labels, 0).nonzero()[0]

    n_Nor = n_exp

    Sub_Nor = sub_score[indx_nor] # Maximum Score for each of abnormal video
    Sub_Abn = sub_score[indx_abn] # Maximum Score for each of normal video

    # Se computa el loss hinge (no entiendo por qué hace el for)
    z = K.ones_like(y_true)
    for i in range(n_Nor):
        sub_z = K.maximum(1 - Sub_Abn + Sub_Nor[i], 0)
        z = K.concatenate([z, K.stack(K.sum(sub_z))])

    z = z[Num_d:]  # We need this step since we have used T.ones_like
    z = K.mean(z, axis=-1) + lamb1 * K.sum(sub_sum_l1) + lamb2 * K.sum(sub_l2)  # Final Loss f

    return z
