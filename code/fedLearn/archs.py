# -*- coding: utf-8 -*-
###############################################################################
# Author: Nicolás Cubero Torres
# Description: Horizontal federated learning simulation utilities implementations
#		for training a deep learning model on a differents federated learning
#		architecture simulated on a single device.
###############################################################################

# Imported modules
from copy import copy
import numpy as np
from tensorflow.keras import Model
from .agr_methods import fedAvg


class FedLearnModel:

	"""Base class representing a Keras Model trainable on an horizontal
		client-server federated learning architecture constructed from a single
		previous Keras Model.

		This model simulates a client-server architecture by considering various
		clients nodes performing the local training and one agregattor server
		node which perform the agregation of model by following a given schedule.

		Parameters
		----------

		build_fn : function
			Function used for building the keras model that is going to be
			fitted

		n_clients : int
			Number of clients nodes to consider for local training
	"""

	def __init__(self, build_fn, n_clients: int, **kwargs):

		self._build_fn = build_fn
		self._n_clients = n_clients

		# Global model
		self._global_model = self._build_fn(**kwargs)

		# The clients model
		self._client_model = dict(zip(range(self._n_clients),
					(self._build_fn(**kwargs) for i in range(self._n_clients))))

		# Check n_clients is correct
		if not isinstance(self._n_clients, int) or self._n_clients <= 0:
			raise ValueError('"n_clients" must be an integer greater than 0')

		# Check global model and clients models are valid Model
		if not isinstance(self._global_model, Model):
			raise ValueError('"build_fn" must return a valid Model object')

		for client in self._client_model:
			if not isinstance(self._client_model[client], Model):
				raise ValueError('"build_fn" must return a valid Model object')

	### Getters ###

	@property
	def global_model(self):
		return self._global_model

	@property
	def client_models(self):
		return self._n_clients

	### Utils functions ###

	def compile(self, **kwargs):

		# Compile the global model
		self._global_model.compile(**kwargs)

		# Compile the clients models
		for client in self._client_model:
			self._client_model[client].compile(**kwargs)

	def fit(self, **kwargs):
		pass

	def evaluate(self, **kwargs):
		# Evaluate the global model
		return self._global_model.evaluate(**kwargs)

	def predict(self, **kwargs):
		# Use the global model for predicting
		return self._global_model.predict(**kwargs)


class SynFedLearnModel(FedLearnModel):

	"""Base class representing a Keras Model trainable on an horizontal syncroned
		client-server federated learning architecture constructed from a single
		previous Keras Model and by considering one agregattor server and various
		clients which performs the local training.
	"""

	def __init__(self, build_fn, n_clients: int, **kwargs):
		super(SynFedLearnModel, self).__init__(build_fn, n_clients, **kwargs)

class SynFedAvgLearnModel(SynFedLearnModel):

	"""Syncroned client-server federated learning architecture implementation
		constructed from a single previous Keras Model and by considering various
		clients which performs the local training and one agregattor server that
		performs agregation through FedAvg method.
	"""

	def __init__(self, build_fn, n_clients: int, **kwargs):
		super(SynFedAvgLearnModel, self).__init__(build_fn, n_clients, **kwargs)

		self.__early_stop = dict() # Early stop schedule

	"""Auxiliar function for copying weights from source model (src_model) to
		the destination model (dst_model)
	"""
	_copy_weights = lambda src_model, dst_model : dst_model.set_weights(src_model.get_weights())

	def __prepare_early_stop_params():

		if self.__early_stop['monitor'] and not isinstance(
													self.__early_stop['monitor'], str):
			raise ValueError('Variable to be monitored by the early stopping schedule'\
								' must be a string')

		if (not isinstance(self.__early_stop['patience'], int) or
													self.__early_stop['patience'] < 0):
			raise ValueError('The patience of the early stopping schedule must be'\
															' an integer greater than 0')

		if (not isinstance(self.__early_stop['delta'], (float, int)) or
														self.__early_stop['delta'] < 0):
			raise ValueError('The delta of early stopping schedule must be grater than 0')

		# Prepare the countage
		self.__early_stop['times'] = {c: 0 for c in self._client_model}

	def fit(self, **kwargs):

		n_epochs = kwargs['epochs'] # Get number of epochs specified
		train = True				# Flag for setting or stopping the training

		# Keep number of samples used by each client node
		samp_per_client = [len((kwargs['x'][c] if kwargs['x'][c] is not None else [])
							if isinstance(kwargs['x'], dict) else kwargs['x']) for c in self._client_model]

		# Get the early stop schedule
		self.__early_stop = dict()
		self.__early_stop['monitor'] = (kwargs['early_stop_monitor'] if
									'early_stop_monitor' in kwargs else None)

		self.__early_stop['patience'] = (kwargs['early_stop_patience'] if
										'early_stop_patience' in kwargs else 5)

		self.__early_stop['delta'] = (kwargs['early_stop_delta'] if
									 	'early_stop_delta' in kwargs else 1e-7)
		self.__early_stop['times'] = {c: 0 for c in self._client_model}

		# Save the history of models
		history = dict() #{c:dict() for c in self._client_model}

		for epoch in range(n_epochs):

			if not train:
				break # Stop training

			if kwargs['verbose']:
				print('Epoch "{}":'.format(epoch))

			# Copy weights from global model to clients models
			for c in self._client_model:
				SynFedAvgLearnModel._copy_weights(self._global_model,
														self._client_model[c])

			# Perform local training
			for c in self._client_model:

				if kwargs['x'][c] is None:
					continue

				# Verbose mode
				if kwargs['verbose']:
					print('Client "{}":'.format(c), end='')

				hist = self._client_model[c].fit(
				    x=kwargs['x'][c] if isinstance(kwargs['x'], dict) else kwargs['x'],
				    y=(kwargs['y'][c] if isinstance(kwargs['y'], dict) else kwargs['y']) if 'y' in kwargs else None,
				    batch_size=kwargs['batch_size'] if 'batch_size' in kwargs else None,
				    epochs=1,
				    verbose=kwargs['verbose'] if 'verbose' in kwargs else 1,
				    callbacks=None,
				    validation_split=kwargs['validation_split'] if 'validation_split' in kwargs else 0.0,
				    validation_data=(kwargs['validation_data'][c] if isinstance(kwargs['validation_data'], dict) else kwargs['validation_data']) if 'validation_data' in kwargs else None,
				    shuffle=kwargs['shuffle'] if 'shuffle' in kwargs else True,
				    class_weight=kwargs['class_weight'] if 'class_weight' in kwargs else None,
				    sample_weight=kwargs['sample_weight'] if 'sample_weight' in kwargs else None,
				    initial_epoch=kwargs['initial_epoch'] if 'initial_epoch' in kwargs else 0,
				    steps_per_epoch=kwargs['steps_per_epoch'] if 'steps_per_epoch' in kwargs else None,
				    validation_steps=kwargs['validation_steps'] if 'validation_steps' in kwargs else None,
				    #validation_batch_size=kwargs['validation_batch_size'] if 'validation_batch_size' in kwargs else None,
				    validation_freq=kwargs['validation_freq'] if 'validation_freq' in kwargs else 1,
				    max_queue_size=kwargs['max_queue_size'] if 'max_queue_size' in kwargs else 10,
				    workers=kwargs['workers'] if 'workers' in kwargs else 1,
				    use_multiprocessing=kwargs['use_multiprocessing'] if 'use_multiprocessing' in kwargs else False
				)

				# Note the metrics into the history

				# Create registry if not exists
				if c not in history:
					history[c] = dict()

					for h in hist.history:
						history[c][h] = np.zeros(n_epochs)

				for h in hist.history:
					history[c][h][epoch] = hist.history[h][0]

				# Execute early stopping schedule
				if (np.abs(history[c][self.__early_stop['monitor']][epoch] -
							history[c][self.__early_stop['monitor']][epoch-1]) <=
													self.__early_stop['delta']):

					self.__early_stop['times'][c] += 1

					if self.__early_stop['times'][c] >= self.__early_stop['patience']:
						train = False # Stop training

					if kwargs['verbose']:
						print('Client {} - After {} epochs without '\
								'improvements, training will be stopped'.format(
											c, self.__early_stop['patience']))
				else:
					self.__early_stop['times'][c] = 0

			# Perform agregation
			fedAvg(models=list(self._client_model.values()),
					samp_per_models=samp_per_client,
					output_model=self._global_model)

		return history
