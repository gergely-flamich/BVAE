import tensorflow as tf
import tensorflow_probability as tfp

from dense_with_prior import GaussianDenseWithGammaPrior

tfd = tfp.distributions
tfl = tf.keras.layers


class DummyBNN(tf.keras.Model):

    def __init__(self,
                 name="dummy_bnn",
                 **kwargs):

        super().__init__(name=name, **kwargs)

        self.transforms = [
            tfl.Dense(units=50,
                      activation=tf.nn.tanh),
            tfl.Dense(units=50,
                      activation=tf.nn.tanh),
            tfl.Dense(units=1)
        ]

        self.likelihood_log_var = tf.Variable(tf.math.log(1e-3), name="likelihood_log_variance")

        self.var_prior = tfd.LogNormal(loc=tf.math.log(1e-4), scale=0.1)

    def build(self, input_shape):

        super().build(input_shape=input_shape)

        self.weight_priors = []

        self.transform_variables = []
        self.num_weight_params = 0

        for layer in self.transforms:
            self.transform_variables += layer.trainable_variables

        for var in self.transform_variables:
            self.num_weight_params += tf.size(var)

            weight_prior = tfd.Independent(distribution=tfd.Normal(loc=tf.zeros_like(var),
                                                                   scale=tf.ones_like(var)),
                                           reinterpreted_batch_ndims=tf.rank(var))

            self.weight_priors.append(weight_prior)

    def weight_prior_log_prob(self):

        prior_log_prob = 0.

        for var, prior in zip(self.transform_variables, self.weight_priors):
            prior_log_prob += prior.log_prob(var)

        return prior_log_prob / tf.cast(self.num_weight_params, tf.float32)

    def get_weights(self):
        return [var.value() for var in self.trainable_variables]

    def set_weights(self, weights):
        for var, weight in zip(self.trainable_variables, weights):
            var.assign(weight)

    def call(self, tensor):

        for layer in self.transforms:
            tensor = layer(tensor)

        return tensor


class MnistBNN(tf.keras.Model):

    def __init__(self,
                 prior_mode,
                 name="bnn",
                 **kwargs):

        super().__init__(name=name, **kwargs)

        self.prior_mode = prior_mode

        self.transforms = [
            tfl.Reshape((28 * 28,)),
            GaussianDenseWithGammaPrior(units=100,
                                        prior_mode=self.prior_mode,
                                        activation=tf.nn.sigmoid),
            GaussianDenseWithGammaPrior(units=10,
                                        prior_mode=self.prior_mode)
        ]

    def resample_weight_prior_parameters(self):

        for layer in self.transforms[1:]:
            layer.resample_precisions()

    def weight_prior_log_prob(self):

        prior_log_prob = 0.
        num_params = 0

        for layer in self.transforms[1:]:
            prior_log_prob += layer.weight_log_prob()
            num_params += layer.num_params

        return prior_log_prob / tf.cast(num_params, tf.float32)

    def hyper_prior_log_prob(self):

        hyperprior_log_prob = self.transforms[1].hyper_prior_log_prob()
        num_params = 0

        for layer in self.transforms[1:]:
            hyperprior_log_prob += layer.hyper_prior_log_prob()
            num_params += layer.num_params

        return hyperprior_log_prob / tf.cast(num_params, tf.float32)

    def get_weights(self):
        return [var.value() for var in self.trainable_variables]

    def set_weights(self, weights):
        for var, weight in zip(self.trainable_variables, weights):
            var.assign(weight)

    def call(self, tensor):

        for layer in self.transforms:
            tensor = layer(tensor)

        return tensor
