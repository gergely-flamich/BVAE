import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfl = tf.keras.layers

class BNN(tf.keras.Model):

    def __init__(self, name="bnn", **kwargs):

        super().__init__(name=name, **kwargs)

        self.weight_priors = []

    def resample_weight_prior_parameters(self):
        """
        Performs a Gibbs step to sample from the hyperposterior
        :return:
        """

        self._hyperprior_log_prob = 0.

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

            self._hyperprior_log_prob = self._hyperprior_log_prob + prec_hyper_dist.log_prob(precision_sample)


    def weight_prior_log_prob(self):

        prior_log_prob = 0.

        for var, weight_prior in zip(self.trainable_variables, self.weight_priors):
            prior_log_prob = prior_log_prob + weight_prior.log_prob(var)

        return prior_log_prob

    def precision_to_scale(self, prec, eps=1e-6):

        return 1. / (tf.sqrt(tf.maximum(prec, eps)) + eps)

    def build(self, input_shape):
        self.transforms = [
            tfl.Reshape((28 * 28,)),
            tfl.Dense(units=100,
                      activation=tf.nn.sigmoid),
            tfl.Dense(units=10,
                      activation=tf.nn.softmax)
        ]

        super().build(input_shape)

        self._weight_prec_hyperprior_concentrations = []
        self._weight_prec_hyperprior_rates = []

        self.weight_precision_hyper_distributions = []

        self.weight_prior_scales = []
        self.weight_priors = []

        self._hyperprior_log_prob = 0.

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

            self._hyperprior_log_prob = self._hyperprior_log_prob + prec_hyperprior.log_prob(precision_sample)

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

