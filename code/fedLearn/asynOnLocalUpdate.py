# -*- coding: utf-8 -*-
###############################################################################
# Author: Nicolás Cubero Torres
# Description: Keras Optimizer implementation of the Asynchronous Online Local
#				Update schedule by using a decay coefficient and the Dynamic
#               Learning Step Size schedule as defined on the following
#               publication:
#
#       Chen, Yujing, Yue Ning, and Huzefa Rangwala. "Asynchronous online
#		    federated learning for edge devices." Chapter (4.2), arXiv preprint
#		     arXiv:1911.02134 (2019)
###############################################################################

# Imported modules
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Optimizer
from tensorflow.python.ops import variables as tf_variables

class AsynOnLocalUpdate(Optimizer):

    """Implementation of the Asynchronous Online Local Update schedule through
        a decay coefficient which contributes to keep a balance between the
		previous global model and the current local model and a Dynamic
        Learning Step Size to set a fairer step size for the strugglers clients

        Attributes
        ----------

        optimizer: Keras Optimizer
            A desired Keras Optimizer used for the base gradient
            updating, e.g. SGD, Adagrad.

            The computation performed by this utility will be directly added to
            the base gradient computation of the optimizer provided.

		global_model_W : list of weights
			List of weights handled from the global model's optimizer. It can be
			got by using the "methods" property of global model's optimizer

		lamb : float, int
			Regularization parameter.

		beta : float, int
			Decay factor used for establishing the balance between the previous
			global model and the current local model
    """
    def __init__(self, optimizer: Optimizer, global_model_W: list,
					lamb: float, beta: float, **kwargs):

        # Check input
        if not isinstance(optimizer, Optimizer):
            raise TypeError('"optimizer" must be a Keras optimizer')

		if not isinstance(global_model_W, list):
			raise TypeError('"global_model" must be a list of weights handled'\
														' by another optimizer')

		if not isinstance(lamb, (float, int)):
			raise TypeError('"lamb" must be float or int greater than 0')

		if lamb <= 0:
			raise ValueError('"lamb" must be greater than 0')

		if not isinstance(beta, (float, int)):
			raise TypeError('"beta" must be a float or integer value')

		if beta <= 0:
			raise ValueError('"beta" must be greater than 0')

        # Declare attributes
        self.__optimizer = optimizer
		self.__global_model_W = global_model_W
		self.__lamb = lamb
		self.__beta = beta

        self.__surr_obj = None          # The surrogate objective function
        self.__pre_surr_obj = None      # Previous surrogate objetive function
        self.__inc_surr_obj = None      # Surrogate Objective function's variation
        self.__pre_inc_surr_obj = None  # Previous surrogate Objective function's variation
		self.__balance = None			# Previous global model and current local model's balance

    def _compute_gradients(self, loss, var_list, grad_loss=None, tape=None):

        # Get the gradients computed over the model variables and process them
        grad_and_vars = self.__optimizer._compute_gradients(loss, var_list,
                                                            grad_loss, tape)
        grads = [g for g, _ in grad_and_vars]
		var = [v for _, v in grad_and_vars]

		reg = self.__calc_regularization()

		# Applied computed regularization to the gradients
		self.__surr_obj = [(g - reg if g is not None else None) for g in grads]

		# Compute variation of the surrogate function
		if not self.__pre_surr_obj:
			self.__inc_surr_obj = self.__surr_obj
			self.__pre_surr_obj = self.__surr_obj
		else:
			self.__inc_surr_obj = [(self.__surr_obj[i] - self.__pre_surr_obj[i]
									if self.__surr_obj[i] is not None else None)
										for i in range(len(self.__surr_obj))]

		# Compute the balance
		beta_rev = (1 - self.__beta)

		if self.__pre_inc_surr_obj:
			if not self.__balance:
				self.__balance = [(beta_rev * self.__pre_inc_surr_obj[i])
							if self.__pre_inc_surr_obj[i] is not None else None
													for i in range(len(grads))]
			else:
				self.__balance = [(self.__beta * self.__balance[i] +
									beta_rev * self.__pre_inc_surr_obj[i])
									if self.__balance[i] is not None else None
													for i in range(len(grads))]


		# Finally, apply the decay coefficient to the gradients
		if not self.__pre_inc_surr_obj and not self.__balance:
			pro_grads = [self.__inc_surr_obj[i]
						if self.__inc_surr_obj[i] is not None else None
											for i in range(len(grads))]
		elif not self.__pre_inc_surr_obj:
			pro_grads = [(self.__inc_surr_obj[i] +
						self.__balance[i])
						if self.__balance[i] is not None else None
						for i in range(len(grads))]
		else:
			pro_grads = [(self.__inc_surr_obj[i] - self.__pre_inc_surr_obj[i] +
					self.__balance[i]) if self.__balance[i] is not None else None
					for i in range(len(grads))]

		self.__pre_inc_surr_obj = self.__inc_surr_obj

		return zip(grads, var)

	def __calc_regularization(self):

		"""The regularization is computed through the following equation:

			reg = (lamb/2)* || wc - wg ||^2

			where:
				wc: client model's weights in which this optimizer will
				be applied

				wg: The global model's weights provided on the global_model
				Model

				lamb: The lambda parameter
		"""
		return (self.__lamb / 2) * np.sum( ((self.weights[i] -
				self.__global_model_W[i])**2).sum() for i in range(
															len(self.weights)))

	def get_config(self):
		config = {
					'lambda': self.__lamb,
					'beta': self.__beta
				}

		config['optimizer'] = super(SGD, self).get_config()

		return config

	## Delegate the rest of Optimizer's functions on the passed Optimizer ##
	def minimize(self, loss, var_list, grad_loss=None, name=None, tape=None):
		return self.__optimizer.minimize(loss, var_list, grad_loss, name, tape)

	def _clip_gradients(self, grads):
		return self.__optimizer._clip_gradients(grads)

	def get_gradients(self, loss, params):
		return self.__optimizer.get_gradients(loss, params) # Revisar

	def apply_gradients(self,
						grads_and_vars,
						name=None,
						experimental_aggregate_gradients=True):
		return self.__optimizer.apply_gradients(
											grads_and_vars,
											name,
											experimental_aggregate_gradients)

	def _aggregate_gradients(self, grads_and_vars):
		return self.__optimizer._aggregate_gradients(grads_and_vars)

	def _distributed_apply(self, distribution, grads_and_vars,
								name, apply_state):
		return self.__optimizer._distributed_apply(distribution, grads_and_vars,
															name, apply_state)

	def get_updates(self, loss, params):
		return self.__optimizer.get_updates(loss, params)

	def _set_hyper(self, name, value):
		self.__optimizer._set_hyper(name, value)

	def _get_hyper(self, name, dtype=None):
		return self.__optimizer._get_hyper(name, dtype)

	def _create_slots(self, var_list):
		self.__optimizer._create_slots(var_list)

	def _create_all_weights(self, var_list):
		self.__optimizer._create_all_weights(var_list)

	def __getattribute__(self, name):
		return self.__optimizer[name]

	def __setattr__(self, name, value):
		self.__optimizer.__setattr[name]= value

	def get_slot_names(self):
		return self.__optimizer.get_slot_names()

	def add_slot(self, var, slot_name, initializer="zeros"):
		return self.__optimizer.add_slot(var, slot_name, initializer)

	def get_slot(self, var, slot_name):
		return self.__optimizer.get_slot(var, slot_name)

	def _prepare(self, var_list):
		return self.__optimizer._prepare(var_list)

	def _prepare_local(self, var_device, var_dtype, apply_state):
		return self.__optimizer._prepare_local(var_device, var_dtype,
																apply_state)

	def _fallback_apply_state(self, var_device, var_dtype):
		return self.__optimizer._fallback_apply_state(var_device, var_dtype)

	def _create_hypers(self):
		self.__optimizer._create_hypers()

	@property
	def iterations(self):
		return self.__optimizer.iterations

	@iterations.setter
	def iterations(self, variable):
		self.__optimizer.iterations = variable

	def _decayed_lr(self, var_dtype):
		return self.__optimizer._decayed_lr(var_dtype)

	def _serialize_hyperparameter(self, hyperparameter_name):
		return self.__optimizer._serialize_hyperparameter(hyperparameter_name)

	def variables(self):
		return self.__optimizer.variables()

	@property
	def weights(self):
		return self.__optimizer.weights

	def get_weights(self):
		return self.__optimizer.get_weights(self)

	def set_weights(self, weights):
		self.__optimizer.set_weights(weights)

	def add_weight(self,
					name,
					shape,
					dtype=None,
					initializer="zeros",
					trainable=None,
					synchronization=tf_variables.VariableSynchronization.AUTO,
					aggregation=tf_variables.VariableAggregation.NONE):

		return self.__optimizer.add_weight(self, name, shape, dtype,
											initializer, trainable,
											synchronization, aggregation)

	def _init_set_name(self, name, zero_based=True):
		self.__optimizer._init_set_name(name, zero_based)

	def _assert_valid_dtypes(self, tensors):
		self.__optimizer._assert_valid_dtypes(tensors)

	def _valid_dtypes(self):
		return self.__optimizer._valid_dtypes()

	def _call_if_callable(self, param):
		return self.__optimizer._call_if_callable(param)

	def _resource_apply_dense(self, grad, handle, apply_state):
		return self.__optimizer._resource_apply_dense(grad, handle, apply_state)

	def _resource_apply_sparse_duplicate_indices(self, grad, handle, indices,
	                                               **kwargs):
		return self.__optimizer._resource_apply_sparse_duplicate_indices(grad,
													handle, indices, **kwargs)

	def _resource_scatter_add(self, x, i, v):
		return self.__optimizer._resource_scatter_add(x, i, v)

	def _resource_scatter_update(self, x, i, v):
		return self.__optimizer._resource_scatter_update(x, i, v)

	@property
	@tracking.cached_per_instance
	def _dense_apply_args(self):
		return self.__optimizer._dense_apply_args

	@property
	@tracking.cached_per_instance
	def _sparse_apply_args(self):
		return self.__optimizer._sparse_apply_args

	def _restore_slot_variable(self, slot_name, variable, slot_variable):
		self.__optimizer._restore_slot_variable(slot_name, variable,
																slot_variable)

	def _create_or_restore_slot_variable(
		self, slot_variable_position, slot_name, variable):
		self.__optimizer._create_or_restore_slot_variable(
	      slot_variable_position, slot_name, variable)
