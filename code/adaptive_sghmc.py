import tensorflow as tf
from tensorflow.python.training import training_ops


class AdaptiveSGHMC(tf.optimizers.Optimizer):
    """
    In this implementation we assume that the scaled gradient noise variance
    (beta_hat in the original paper) is set to 0.
    """

    def __init__(self,
                 learning_rate,
                 burnin,
                 data_size=1,
                 momentum_decay=0.01,
                 name="AdaptiveSGHMC",
                 **kwargs):

        with tf.name_scope(name):
            self._learning_rate = tf.convert_to_tensor(learning_rate, name="learning_rate")
            self._burnin = tf.convert_to_tensor(burnin, name="burnin", dtype=tf.int64)
            self._data_size = tf.convert_to_tensor(data_size, name="data_size", dtype=tf.int64)
            self._momentum_decay = tf.convert_to_tensor(momentum_decay, name="momentum_decay", )

            self._friction = tf.Variable(0., name="friction")

            super().__init__(name=name, **kwargs)

    def get_config(self):
        pass

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "momentum")

            # V_hat
            self.add_slot(var, "squared_grad_magnitude", initializer="ones")

            # g
            self.add_slot(var, "smoothed_gradient", initializer="ones")

            # tau
            self.add_slot(var, "exponential_average_coeff", initializer="ones")

    def _resource_apply_dense(self, grad, var, apply_state=None):

        momentum = self.get_slot(var, "momentum")
        squared_grad_magnitude = self.get_slot(var, "squared_grad_magnitude")
        smoothed_gradient = self.get_slot(var, "smoothed_gradient")
        exponential_average_coeff = self.get_slot(var, "exponential_average_coeff")

        return self._sghmc_step(grad=grad,
                                variable=var,
                                momentum=momentum,
                                squared_grad_magnitude=squared_grad_magnitude,
                                smoothed_gradient=smoothed_gradient,
                                exponential_average_coeff=exponential_average_coeff)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):

        momentum = self.get_slot(var, "momentum")
        squared_grad_magnitude = self.get_slot(var, "squared_grad_magnitude")
        smoothed_gradient = self.get_slot(var, "smoothed_gradient")
        exponential_average_coeff = self.get_slot(var, "exponential_average_coeff")

        return self._sghmc_step(grad=grad,
                                variable=var,
                                momentum=momentum,
                                squared_grad_magnitude=squared_grad_magnitude,
                                smoothed_gradient=smoothed_gradient,
                                exponential_average_coeff=exponential_average_coeff,
                                indices=indices)

    def _sghmc_step(self,
                    grad,
                    variable,
                    momentum,
                    squared_grad_magnitude,
                    smoothed_gradient,
                    exponential_average_coeff,
                    epsilon=1e-16,
                    indices=None):

        # Scale the gradient according to the data size
        scaled_grad = grad * tf.cast(self._data_size, grad.dtype)

        # If we are still in the burn-in phase, adapt:
        # the preconditioner,
        # the the exponentail averaging coefficient
        # the gradient noise
        deltas = tf.cond(self.iterations < self._burnin,
                         true_fn=lambda: self._burnin_hyperparameter_deltas(scaled_grad=scaled_grad,
                                                                            exponential_average_coeff=exponential_average_coeff,
                                                                            smoothed_gradient=smoothed_gradient,
                                                                            squared_grad_magnitude=squared_grad_magnitude,
                                                                            epsilon=epsilon),
                         false_fn=lambda: (0., 0., 0.))

        delta_ea_coeff, delta_smoothed_gradient, delta_squared_grad_magnitude = deltas

        new_ea = exponential_average_coeff + delta_ea_coeff
        new_smoothed_grad = smoothed_gradient + delta_smoothed_gradient
        new_squared_grad_magnitude = squared_grad_magnitude + delta_squared_grad_magnitude

        exponential_average_coeff.assign(new_ea)
        smoothed_gradient.assign(new_smoothed_grad)
        squared_grad_magnitude.assign(new_squared_grad_magnitude)

        # ----------------------------
        # Actual SGHMC step
        # ----------------------------

        grad_magnitude = tf.sqrt(squared_grad_magnitude)

        preconditioner = 1. / (grad_magnitude + epsilon)

        # Note the assumption that momentum_decay = learning_rate * precondtioner^-1 * C
        noise_variance = 2. * preconditioner * self._momentum_decay - self._learning_rate ** 2
        noise_stddev = self._learning_rate * tf.sqrt(tf.maximum(noise_variance, 1e-16))

        momentum_noise = tf.random.normal(shape=scaled_grad.shape, dtype=scaled_grad.dtype)
        momentum_noise = noise_stddev * momentum_noise

        momentum_delta = -self._momentum_decay * momentum + \
                         -self._learning_rate ** 2 * preconditioner * scaled_grad + \
                         momentum_noise

        new_variable = variable + momentum
        new_momentum = momentum + momentum_delta

        # Dense moment update
        if indices is None:
            # Note the minus sign on the delta argument. This is because we wish to perform gradient ascent.
            variable.assign(new_variable)
            momentum.assign(new_momentum)

        else:
            self._resource_scatter_update(variable, indices, new_variable)
            self._resource_scatter_update(momentum, indices, new_momentum)

        # https://github.com/tensorflow/tensorflow/issues/30711#issuecomment-512921409
        return []

    def _burnin_hyperparameter_deltas(self,
                                      scaled_grad,
                                      exponential_average_coeff,
                                      smoothed_gradient,
                                      squared_grad_magnitude,
                                      epsilon=1e-16):

        ea_coeff_inv = 1. / (exponential_average_coeff + 1.)

        # Tau delta
        delta_ea_coeff = tf.square(smoothed_gradient) / (squared_grad_magnitude + epsilon)
        delta_ea_coeff = -exponential_average_coeff * delta_ea_coeff + 1.

        # g delta
        delta_smoothed_gradient = ea_coeff_inv * (-smoothed_gradient + scaled_grad)

        # Preconditioner (V_theta) delta
        delta_squared_grad_magnitude = ea_coeff_inv * (-squared_grad_magnitude + tf.square(scaled_grad))

        return delta_ea_coeff, delta_smoothed_gradient, delta_squared_grad_magnitude
