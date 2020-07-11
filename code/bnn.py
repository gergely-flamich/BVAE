import tensorflow as tf
import tensorflow_probability as tfp

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
                 name="bnn",
                 **kwargs):

        super().__init__(name=name, **kwargs)

        self.transforms = [
            tfl.Reshape((28 * 28,)),
            tfl.Dense(units=100,
                      activation=tf.nn.sigmoid),
            tfl.Dense(units=10,
                      activation=tf.nn.softmax)
        ]

        self.weight_priors = []


    def resample_weight_prior_parameters(self):
        """
        Performs a Gibbs step to sample from the hyperposterior
        :return:
        """

        # Update the scale distributions
        for var, concentration, rate, prec_hyper_dist, weight_prior_scale in zip(self.trainable_variables,
                                                                                 self._weight_prec_hyperprior_concentrations,
                                                                                 self._weight_prec_hyperprior_rates,
                                                                                 self.weight_precision_hyper_distributions,
                                                                                 self.weight_prior_scales):
            concentration.assign(concentration + 0.5)
            rate.assign(rate + 0.5 * tf.square(var))

            precision_sample = prec_hyper_dist.sample()

            weight_prior_scale.assign(self.precision_to_scale(precision_sample))

    def weight_prior_log_prob(self):

        prior_log_prob = 0.

        for var, weight_prior in zip(self.trainable_variables, self.weight_priors):
            prior_log_prob = prior_log_prob + weight_prior.log_prob(var)

        return prior_log_prob

    def hyperprior_log_prob(self):

        hyperprior_log_prob = 0.

        for stddev, precision_prior in zip(self.weight_prior_scales, self.weight_precision_hyper_distributions):
            hyperprior_log_prob += precision_prior.log_prob(1. / tf.square(stddev))

        return hyperprior_log_prob

    def precision_to_scale(self, prec, eps=1e-6):

        return 1. / (tf.sqrt(tf.maximum(prec, eps)) + eps)

    def get_weights(self):
        return [var.value() for var in self.trainable_variables]

    def set_weights(self, weights):
        for var, weight in zip(self.trainable_variables, weights):
            var.assign(weight)

    def build(self, input_shape):

        super().build(input_shape)

        self._weight_prec_hyperprior_concentrations = []
        self._weight_prec_hyperprior_rates = []

        self.weight_precision_hyper_distributions = []

        self.weight_prior_scales = []
        self.weight_priors = []

        for var in self.trainable_variables:
            concentration = tf.Variable(tf.ones_like(var),
                                        name=f"{var.name}/prec_hyperprior_concentration",
                                        trainable=False)

            rate = tf.Variable(tf.ones_like(var),
                               name=f"{var.name}/prec_hyperprior_rate",
                               trainable=False)

            prec_hyperprior = tfd.Independent(distribution=tfd.Gamma(concentration=concentration, rate=rate),
                                              reinterpreted_batch_ndims=tf.rank(concentration))

            precision_sample = prec_hyperprior.sample()

            weight_prior_scale = tf.Variable(self.precision_to_scale(precision_sample),
                                             name=f"{var.name}/prior_scale",
                                             trainable=False)

            weight_prior = tfd.Independent(distribution=tfd.Normal(loc=0., scale=weight_prior_scale),
                                           reinterpreted_batch_ndims=tf.rank(concentration))

            self._weight_prec_hyperprior_concentrations.append(concentration)
            self._weight_prec_hyperprior_rates.append(rate)
            self.weight_precision_hyper_distributions.append(prec_hyperprior)

            self.weight_prior_scales.append(weight_prior_scale)
            self.weight_priors.append(weight_prior)

    def call(self, inputs, training=None, mask=None):

        tensor = inputs

        for layer in self.transforms:
            tensor = layer(tensor)

        return tensor

