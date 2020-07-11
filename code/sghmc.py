import tensorflow as tf
from tensorflow.python.training import training_ops


class SGHMC(tf.optimizers.Optimizer):
    """
    In this implementation we assume that the scaled gradient noise variance
    (beta_hat in the original paper) is set to 0.
    """

    def __init__(self,
                 learning_rate,
                 data_size=1,
                 momentum_decay=0.01,
                 name="SGHMC",
                 **kwargs):

        with tf.name_scope(name):
            self._learning_rate = tf.convert_to_tensor(learning_rate, name="learning_rate")
            self._data_size = tf.convert_to_tensor(data_size, name="data_size", dtype=tf.int64)
            self._momentum_decay = tf.convert_to_tensor(momentum_decay, name="momentum_decay", )

            super().__init__(name=name, **kwargs)

    def get_config(self):
        pass

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "momentum")

    def _resource_apply_dense(self, grad, var, apply_state=None):

        momentum = self.get_slot(var, "momentum")

        return self._sghmc_step(grad=grad,
                                variable=var,
                                momentum=momentum)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):

        momentum = self.get_slot(var, "momentum")

        return self._sghmc_step(grad=grad,
                                variable=var,
                                momentum=momentum,
                                indices=indices)

    def _sghmc_step(self, grad, variable, momentum, indices=None):

        # Scale the gradient according to the data size
        scaled_grad = grad * tf.cast(self._data_size, grad.dtype)

        noise_stddev = tf.sqrt(2. * self._learning_rate * self._momentum_decay)
        momentum_noise = tf.random.normal(shape=scaled_grad.shape, dtype=scaled_grad.dtype)
        momentum_noise = noise_stddev * momentum_noise

        momentum_delta = -self._momentum_decay * momentum + \
                         -self._learning_rate * scaled_grad + \
                         momentum_noise

        new_variable = variable + momentum
        new_momentum = momentum + momentum_delta

        # Dense moment update
        if indices is None:
            # Note the minus sign on the delta argument. This is because we wish to perform gradient ascent.
            update_ops = [
                variable.assign(new_variable).op,
                momentum.assign(new_momentum).op
            ]

        else:
            update_ops = [
                self._resource_scatter_update(variable, indices, new_variable),
                self._resource_scatter_update(momentum, indices, new_momentum)
            ]

        return tf.group(update_ops)
