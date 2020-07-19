import tensorflow as tf
from functools import partial


class AdaptiveSGHMC(tf.optimizers.Optimizer):
    """
    In this implementation we assume that the scaled gradient noise variance
    (beta_hat in the original paper) is set to 0.
    """

    _VELOCITY_NAME = "velocity"
    _SQUARED_GRAD_NAME = "squared_grad"
    _SMOOTH_GRAD_NAME = "smooth_grad"
    _TAU_NAME = "tau"

    def __init__(self,
                 learning_rate,
                 burnin,
                 initialization_rounds=0,
                 overestimation_rate=1.,
                 data_size=1,
                 momentum_decay=0.01,
                 name="AdaptiveSGHMC",
                 **kwargs):

        with tf.name_scope(name):
            self._learning_rate = tf.convert_to_tensor(learning_rate, name="learning_rate")
            self._burnin = tf.convert_to_tensor(burnin, name="burnin", dtype=tf.int64)
            self._data_size = tf.convert_to_tensor(data_size, name="data_size", dtype=tf.int64)
            self._friction = tf.convert_to_tensor(momentum_decay, name="momentum_decay", )

            self._initialization_rounds = tf.convert_to_tensor(initialization_rounds,
                                                               name="initialization_rounds",
                                                               dtype=tf.int64)

            self._overestimation_rate = tf.convert_to_tensor(overestimation_rate,
                                                             name="overestimation_rate",
                                                             dtype=tf.float32)

            self._learning_rate_squared = self._learning_rate ** 2.

            self.eps = 1e-6
            self.eps_squared = self.eps ** 2.

            super().__init__(name=name, **kwargs)

    def get_config(self):
        pass

    def _create_slots(self, var_list):

        # If we estimate the auxiliary parameters before the burn-in, we initialize them to zeros,
        # if there is not pre-burn-in initialization, ones are usually reasonable
        auxiliary_slot_initializer = "zeros" if self._initialization_rounds > 0 else "ones"

        for var in var_list:
            self.add_slot(var, self._VELOCITY_NAME, initializer="zeros")

            # V_hat
            self.add_slot(var, self._SQUARED_GRAD_NAME, initializer=auxiliary_slot_initializer)

            # g
            self.add_slot(var, self._SMOOTH_GRAD_NAME, initializer=auxiliary_slot_initializer)

            # exponential average coefficient
            self.add_slot(var, self._TAU_NAME, initializer="ones")

    def _resource_apply_dense(self, grad, var, apply_state=None):

        velocity = self.get_slot(var, self._VELOCITY_NAME)
        squared_grad = self.get_slot(var, self._SQUARED_GRAD_NAME)
        smooth_grad = self.get_slot(var, self._SMOOTH_GRAD_NAME)
        tau = self.get_slot(var, self._TAU_NAME)

        return self._sghmc_step(batch_grad=grad,
                                variable=var,
                                velocity=velocity,
                                squared_grad=squared_grad,
                                smooth_grad=smooth_grad,
                                tau=tau)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):

        momentum = self.get_slot(var, self._VELOCITY_NAME)
        squared_grad = self.get_slot(var, self._SQUARED_GRAD_NAME)
        smooth_grad = self.get_slot(var, self._SMOOTH_GRAD_NAME)
        tau = self.get_slot(var, self._TAU_NAME)

        return self._sghmc_step(batch_grad=grad,
                                variable=var,
                                velocity=momentum,
                                squared_grad=squared_grad,
                                smooth_grad=smooth_grad,
                                tau=tau,
                                indices=indices)

    @tf.function(experimental_relax_shapes=True)
    def _sghmc_step(self,
                    batch_grad,
                    variable,
                    velocity,
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
        if self.iterations < self._initialization_rounds:
            smooth_grad.assign_add(batch_grad)
            squared_grad.assign_add(tf.square(batch_grad))

            # During the initialization rounds, there are no updates to the actual
            # parameters and momentum
            return []

        if self._initialization_rounds > 0 and self.iterations == self._initialization_rounds:
            n0 = tf.cast(self._initialization_rounds, batch_grad.dtype)
            c = tf.cast(self._overestimation_rate, batch_grad.dtype)

            tf.print("SGHMC hyperparameters initialized!") #, tf.reduce_max(tf.square(smooth_grad / n0) - squared_grad / n0 * c))

            smooth_grad.assign(smooth_grad / n0)
            squared_grad.assign((squared_grad / n0) * c)
            tau.assign(n0 * tf.ones_like(tau))

        # ---------------------------------------------------------------------
        # Optimizer hyper-parameter adaptation during burn-in
        # ---------------------------------------------------------------------
        # If we are still in the burn-in phase, adapt:
        # the preconditioner,
        # the the exponentail averaging coefficient
        # the gradient noise

        if self._initialization_rounds < self.iterations and self.iterations < self._burnin:

            noise_variance_ratio = tf.square(smooth_grad) / (squared_grad + self.eps)
            #noise_variance_ratio = tf.minimum(noise_variance_ratio, 1.)

            # Tau delta
            delta_tau = -tau * noise_variance_ratio + 1.

            tau_inv = 1. / (tau + self.eps)

            # g delta
            delta_smooth_grad = tau_inv * (-smooth_grad + batch_grad)

            # V_theta delta
            delta_squared_grad = tau_inv * (-squared_grad + tf.square(batch_grad))

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
        inverse_mass = 1. / tf.sqrt(squared_grad + self.eps_squared)

        # Note the assumption that momentum_decay = learning_rate * precondtioner^-1 * C
        noise_variance = 2. * inverse_mass * self._friction - self._learning_rate_squared
        noise_variance = self._learning_rate_squared * tf.maximum(noise_variance, self.eps_squared)

        velocity_noise = tf.random.normal(shape=batch_grad.shape, dtype=batch_grad.dtype)
        velocity_noise = tf.sqrt(noise_variance) * velocity_noise

        velocity_delta = -self._friction * velocity + \
                         -self._learning_rate_squared * inverse_mass * batch_grad + \
                         velocity_noise

        new_velocity = velocity + velocity_delta
        new_variable = variable + new_velocity

        if indices is None:
            # Note the minus sign on the delta argument. This is because we wish to perform gradient ascent.
            variable.assign(new_variable)
            velocity.assign(new_velocity)

        else:
            self._resource_scatter_update(variable, indices, new_variable)
            self._resource_scatter_update(velocity, indices, new_velocity)

        # https://github.com/tensorflow/tensorflow/issues/30711#issuecomment-512921409
        return []
