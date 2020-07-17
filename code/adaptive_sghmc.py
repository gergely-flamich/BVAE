import tensorflow as tf
from functools import partial


class AdaptiveSGHMC(tf.optimizers.Optimizer):
    """
    In this implementation we assume that the scaled gradient noise variance
    (beta_hat in the original paper) is set to 0.
    """

    _MOMENTUM_NAME = "momentum"
    _SQUARED_GRAD_NAME = "squared_grad"
    _SMOOTH_GRAD_NAME = "smooth_grad"
    _TAU_NAME = "tau"

    def __init__(self,
                 learning_rate,
                 burnin,
                 initialization_rounds=10,
                 overestimation_rate=1000.,
                 data_size=1,
                 momentum_decay=0.01,
                 name="AdaptiveSGHMC",
                 **kwargs):

        with tf.name_scope(name):
            self._learning_rate = tf.convert_to_tensor(learning_rate, name="learning_rate")
            self._burnin = tf.convert_to_tensor(burnin, name="burnin", dtype=tf.int64)
            self._data_size = tf.convert_to_tensor(data_size, name="data_size", dtype=tf.int64)
            self._momentum_decay = tf.convert_to_tensor(momentum_decay, name="momentum_decay", )

            self._initialization_rounds = tf.convert_to_tensor(initialization_rounds,
                                                               name="initialization_rounds",
                                                               dtype=tf.int64)

            self._overestimation_rate = tf.convert_to_tensor(overestimation_rate,
                                                             name="overestimation_rate",
                                                             dtype=tf.float32)

            self._learning_rate_squared = self._learning_rate ** 2.

            self.eps = 1e-6

            super().__init__(name=name, **kwargs)

    def get_config(self):
        pass

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, self._MOMENTUM_NAME, initializer="zeros")

            # V_hat
            self.add_slot(var, self._SQUARED_GRAD_NAME, initializer="ones")

            # g
            self.add_slot(var, self._SMOOTH_GRAD_NAME, initializer="ones")

            # exponential average coefficient
            self.add_slot(var, self._TAU_NAME, initializer="ones")

    def _resource_apply_dense(self, grad, var, apply_state=None):

        momentum = self.get_slot(var, self._MOMENTUM_NAME)
        squared_grad = self.get_slot(var, self._SQUARED_GRAD_NAME)
        smooth_grad = self.get_slot(var, self._SMOOTH_GRAD_NAME)
        tau = self.get_slot(var, self._TAU_NAME)

        return self._sghmc_step(batch_grad=grad,
                                variable=var,
                                momentum=momentum,
                                squared_grad=squared_grad,
                                smooth_grad=smooth_grad,
                                tau=tau)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):

        momentum = self.get_slot(var, self._MOMENTUM_NAME)
        squared_grad = self.get_slot(var, self._SQUARED_GRAD_NAME)
        smooth_grad = self.get_slot(var, self._SMOOTH_GRAD_NAME)
        tau = self.get_slot(var, self._TAU_NAME)

        return self._sghmc_step(batch_grad=grad,
                                variable=var,
                                momentum=momentum,
                                squared_grad=squared_grad,
                                smooth_grad=smooth_grad,
                                tau=tau,
                                indices=indices)

    def _sghmc_step(self,
                    batch_grad,
                    variable,
                    momentum,
                    squared_grad,
                    smooth_grad,
                    tau,
                    indices=None):

        # Scale the gradient according to the data size
        batch_grad = batch_grad * tf.cast(self._data_size, batch_grad.dtype)

        # ---------------------------------------------------------------------
        # Hyper-parameter adaptation based on the paper "No more pesky learning rates"
        # ---------------------------------------------------------------------

        # Compute average over the first few iterations
        # if self.iterations < self._initialization_rounds:
        #     smooth_grad.assign_add(batch_grad)
        #     squared_grad.assign_add(tf.square(batch_grad))

        update_val = tf.cond(self.iterations < self._initialization_rounds,
                             true_fn=lambda: batch_grad,
                             false_fn=lambda: 0.)

        init_smooth_grad = smooth_grad + update_val
        init_squared_grad = squared_grad + tf.square(update_val)

        smooth_grad.assign(init_smooth_grad)
        squared_grad.assign(init_squared_grad)

        def average_initial_vals():
            n0 = tf.cast(self._initialization_rounds, batch_grad.dtype)
            c = tf.cast(self._overestimation_rate, batch_grad.dtype)

            tf.print("SGHMC hyperparameters initialized!") #, tf.reduce_max(tf.square(smooth_grad / n0) - squared_grad / n0 * c))

            return (
                smooth_grad / n0,
                (squared_grad / n0) * c,
                n0 * tf.ones_like(tau)
            )

        init_vals = tf.cond(self.iterations == self._initialization_rounds,
                            true_fn=average_initial_vals,
                            false_fn=lambda: (smooth_grad, squared_grad, tau))

        init_smooth_grad, init_squared_grad, init_tau = init_vals

        smooth_grad.assign(init_smooth_grad)
        squared_grad.assign(init_squared_grad)
        tau.assign(init_tau)

        # ---------------------------------------------------------------------
        # Optimizer hyper-parameter adaptation during burn-in
        # ---------------------------------------------------------------------
        # If we are still in the burn-in phase, adapt:
        # the preconditioner,
        # the the exponentail averaging coefficient
        # the gradient noise
        calculate_deltas = partial(self._burnin_hyperparameter_deltas,
                                   batch_grad=batch_grad,
                                   tau=tau,
                                   smooth_grad=smooth_grad,
                                   squared_grad=squared_grad)

        deltas = tf.cond(tf.logical_and(self._initialization_rounds < self.iterations,
                                        self.iterations < self._burnin),

                         true_fn=calculate_deltas,

                         false_fn=lambda: (0., 0., 0.))

        delta_tau, delta_smooth_grad, delta_squared_grad = deltas

        new_tau = tau + delta_tau
        new_smooth_grad = smooth_grad + delta_smooth_grad
        new_squared_grad = squared_grad + delta_squared_grad

        # Simultaneous update to optimizer hyper-parameters
        tau.assign(new_tau)
        smooth_grad.assign(new_smooth_grad)
        squared_grad.assign(new_squared_grad)

        # ---------------------------------------------------------------------
        # Actual SGHMC step
        # ---------------------------------------------------------------------
        preconditioner = 1. / tf.sqrt(squared_grad + 1e-16)

        # Note the assumption that momentum_decay = learning_rate * precondtioner^-1 * C
        noise_variance = 2. * preconditioner * self._momentum_decay - self._learning_rate_squared
        noise_variance = self._learning_rate * tf.maximum(noise_variance, 1e-16)

        noise_stddev = tf.sqrt(noise_variance)

        momentum_noise = tf.random.normal(shape=batch_grad.shape, dtype=batch_grad.dtype)
        momentum_noise = noise_stddev * momentum_noise

        momentum_delta = -self._momentum_decay * momentum + \
                         -self._learning_rate_squared * preconditioner * batch_grad + \
                         momentum_noise

        new_momentum = momentum + momentum_delta
        new_variable = variable + new_momentum

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
                                      batch_grad,
                                      tau,
                                      smooth_grad,
                                      squared_grad,
                                      tau_eps=1e-6):

        # tf.print("Iter:", self.iterations, "Min EA:", tf.reduce_min(tau),
        #          "Max SG:", tf.reduce_max(smooth_grad), "Max SGM: ", tf.reduce_max(squared_grad), "Grad",
        #          tf.reduce_max(batch_grad))

        # Tau delta
        delta_tau = -tau * (tf.square(smooth_grad) / (squared_grad + self.eps)) + 1.

        tau_inv = 1. / (tau + tau_eps)

        # g delta
        delta_smooth_grad = tau_inv * (-smooth_grad + batch_grad)

        # V_theta delta
        delta_squared_grad = tau_inv * (-squared_grad + tf.square(batch_grad))

        return delta_tau, delta_smooth_grad, delta_squared_grad
