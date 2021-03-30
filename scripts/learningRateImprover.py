# -*- coding:utf-8 -*-
from collections import deque

from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback

"""
Clase LearningRateImprover
Descripción: Callback que permite estimar y ajustar el mejor valor del parámetro de
        learning rate para acelerar el proceso de aprendizaje
        mediante la minimización de un parámetro del proceso de
        aprendizaje que se deba de minimizar evitando caer en una
        redución casi nula

Parámetros:
        parameter: String con el nombre del parámetro a minimizar
        initial_value: Valor inicial del parámetro de aprendizaje
        min_lr: Valor tal que si la reducción cae por debajo de él
            no se sigue descenciendo el learning rate
        factor: Factor de descenso de learning rate
        patience: Número de epochs a transcurrir hasta que varíe el learning rate
        comparación: Factor usado para comparar si la variación del valor medido decae or debajo de 0.
        stop_minimum: Determina si el entrenamiento se debe de detener cuando el learning rate alcanza épsilon (true) o no (false)
Raises:
        ValueError si initial_value está fuera de (0,1]
        ValueError si min_lr es inferior o igual a 0
Atención:    No usar ningún optimizador que ya modifique el learning rate
"""


class LearningRateImprover(Callback):

    def __init__(self, parameter='val_loss', minimize=True, initial_value=None,
                 min_lr=1e-7, factor=0.9, patience=5, min_delta=1e-6,
                 verbose=0, stop_minimum=True, restore_best_weights=False,
                 acumulate_epochs=False):
        super(LearningRateImprover, self).__init__()

        # Check input
        if not isinstance(min_lr, (float, int)):
            raise TypeError('The minimum learning rate must be float')

        if min_lr <= 0:
            raise ValueError('The minimum learning rate must be greater than 0')

        if initial_value and not isinstance(initial_value, (float, int)):
            raise TypeError('initial_value must be float or int')

        if initial_value and initial_value <= 0:
            raise ValueError('initial_value must be greater than 0')

        if not isinstance(acumulate_epochs, bool):
            raise TypeError('"acumulate_epochs" must be bool')

        self._parameter = parameter
        self._min_lr = min_lr
        self._initial_value = initial_value
        self._factor = factor
        self._patience = patience
        self._min_delta = min_delta
        self._verbose = verbose
        self._minimize = minimize
        self._stop_minimum = stop_minimum
        self._restore_best_weights = restore_best_weights
        self._acumulate_epochs = acumulate_epochs

        self._prev_value = None
        self._val_param = deque(maxlen=patience)
        self._best_weights = None

        self._stop_training = False  # Wheter the training is stopped

        self._lr_history = deque()  # Learning rate values' history

    # self._global_epochs = 0 # Global number of epoch if acumulate_epochs is set

    def on_train_begin(self, logs=None):

        self._stop_training = False

        # Reset epochs and lr history if not epochs won't be acumulated
        if not self._acumulate_epochs:
            self._lr_history.clear()

        if self._initial_value and (not self._acumulate_epochs or
                                    not self._lr_history):
            K.set_value(self.model.optimizer.lr, self._initial_value)
        else:
            self._initial_value = float(K.get_value(self.model.optimizer.lr))

    def on_epoch_end(self, epoch, logs=None):
        # def on_batch_end(self,epoch,logs=None):

        epoch = len(self._lr_history)  # True number of epochs registered

        if not logs:
            return

        if self._parameter not in logs:
            raise ValueError(f'{self._parameter} is unknow. Allowed parameters:{logs.keys()}')

        value = logs.get(self._parameter)
        lr = float(K.get_value(self.model.optimizer.lr))

        # Store the current variation of the desired parameter
        if epoch > 0:
            self._val_param.append(value - self._prev_value)

            if self._verbose == 1:
                print(f'Variation of {self._parameter} is {self._val_param[-1]}')

        self._lr_history.append(lr)
        self._prev_value = value

        # Perform learning rate readjust if proceed
        if len(self._val_param) != self._patience:
            return

        avg_delta = (sum(self._val_param) / len(self._val_param))

        # Average delta parameter reduction is lower than minimum
        # delta and learning rate is readjusted
        if avg_delta > -self._min_delta:
            if lr > self._min_lr:
                lr *= self._factor
                K.set_value(self.model.optimizer.lr, lr)

                self._val_param.clear()

            elif self._stop_minimum:
                # Stop Training
                if self._verbose:
                    print('Training has stacked into a local minimum and it\'s going to be stopped')

                self.model.stop_training = True
                self._stop_training = True

                # Restore best weights
                if self._restore_best_weights and self._best_weights is not None:
                    self.model.set_weights(self._best_weights)

            if self._verbose:
                print(f' .Average of last {self._patience} epochs variations of {self._parameter} is {avg_delta}')
                print(f'Epoch #{epoch} - current value of learning rate {lr}')

        elif self._restore_best_weights:
            self._best_weights = self.model.get_weights()

        # Se vacía la lista
        # self._val_param = self._val_param[1:]

    @property
    def lr_history(self):
        return tuple(self._lr_history)

    @property
    def stop_training(self):
        return self._stop_training
