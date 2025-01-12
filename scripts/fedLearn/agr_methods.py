# -*- coding: utf-8 -*-
###############################################################################
# Author: Nicolás Cubero Torres
# Description: Agregattion methods implementation for federated learning
#				architectures..
###############################################################################

# Imported modules
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer

def fedAvg(models: list, samp_per_models: list, output_model: Model):

	# Check input parameters
	if (not isinstance(models, list) or
							any(not isinstance(x, Model) for x in models)):
		raise ValueError('"models" should be a list of keras models')

	if (not isinstance(samp_per_models, list) or
				any(not isinstance(x, int) for x in samp_per_models) or
				len(models) != len(samp_per_models)):
		raise ValueError('"samp_per_models" should be a list of int with'\
						' the same length as "models"')

	if any(x < 0 for x in samp_per_models):
		raise ValueError('"samp_per_models" values can\'t be less than 0')

	if not isinstance(output_model, Model):
		raise ValueError('"output_model" must be a Model')

	# Processing
	total_samples = float(sum(samp_per_models))

	# Agregates weights of first client
	for l in range(len(output_model.layers)):
		output_model.get_layer(index=l).set_weights(
				[w*(samp_per_models[0]/total_samples) for w in models[0].get_layer(
															index=l).get_weights()]
		)

	# Agregates weights of the rest clients
	for c in range(1, len(models)):

		if samp_per_models[c] > 0:

			rate = (samp_per_models[c]/total_samples)

			for l in range(len(output_model.layers)):
				w_length = len(models[c].get_layer(index=l).get_weights())

				output_model.get_layer(index=l).set_weights(
					[output_model.get_layer(index=l).get_weights()[i] +
						models[c].get_layer(index=l).get_weights()[i] *
						rate for i in range(w_length)]
				)


def asyncUpd(global_model: Model, client_models: list, pre_client_models: list,
				samp_per_models: list, output_model: Model):

	# Check input parameters
	if not isinstance(global_model, Model):
		raise ValueError('"global_model" must be a Keras model')

	if (not isinstance(client_models, list) or
						any(not isinstance(x, Model) for x in client_models)):
		raise ValueError('"clients_models" should be a list of keras models')

	if (not isinstance(pre_client_models, list) or
					any(not isinstance(x, Model) for x in pre_client_models)):
		raise ValueError('"pre_client_models" should be a list of keras models')

	if len(client_models) != len(pre_client_models):
		raise ValueError('"client_models" and "pre_client_models" lists must '\
						'have the same length')

	if (not isinstance(samp_per_models, list) or
				any(not isinstance(x, int) for x in samp_per_models) or
				len(client_models) != len(samp_per_models)):
		raise ValueError('"samp_per_models" should be a list of int with'\
						' the same length as "models"')

	if any(x < 0 for x in samp_per_models):
		raise ValueError('"samp_per_models" values must be greater than 0')

	if not isinstance(output_model, Model):
		raise ValueError('"output_model" must be a Model')

	# Processing
	total_samples = float(sum(samp_per_models))

	# Agregates weights of clients over the output model
	for c in range(len(client_models)):

		for l in range(len(output_model.layers)):
			w_length = len(client_models[c].get_layer(index=l).get_weights())
			rate = (samp_per_models[c]/total_samples)

			output_model.get_layer(index=l).set_weights(
				[output_model.get_layer(index=l).get_weights()[i] -
					(pre_client_models[c].get_layer(index=l).get_weights()[i] -
					client_models[c].get_layer(index=l).get_weights()[i]) *
					rate for i in range(w_length)]
			)

def globFeatRep(layer: Layer):

	"""Applies a Global Feature Representation Learning on a given
		Keras layer

		Params
		------
		layer : tf.keras.Layer
			Layer in whichi the global feature representation learning will be
			applied
	"""

	## Compute alpha
	layer_wghts, layer_bias = layer.get_weights()

	exp_alpha_wghts = np.exp(layer_wghts)
	sum_exp = np.sum(exp_alpha_wghts, axis=-1)

	alpha_wghts = exp_alpha_wghts / np.expand_dims(sum_exp, axis=-1)

	## Compute the global feature representation of layer
	layer.set_weights([layer_wghts*alpha_wghts, layer_bias])
