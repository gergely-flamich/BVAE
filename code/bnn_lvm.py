import tensorflow as tf
import tensorflow_probability as tfp

from dense_with_prior import GaussianDenseWithGammaPrior

tfd = tfp.distributions
tfl = tf.keras.layers


class Encoder(tfl.Layer):

    def __init__(self, latent_dim, name="encoder", *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

        self.latent_dim = latent_dim

    def build(self, input_shape):
        self.transforms = [
            tfl.Reshape(target_shape=(28 * 28,)),
            GaussianDenseWithGammaPrior(units=1024,
                                        prior_mode="weight_and_bias",
                                        activation=tf.nn.relu),
            GaussianDenseWithGammaPrior(units=512,
                                        prior_mode="weight_and_bias",
                                        activation=tf.nn.relu),
        ]

        self.loc_head = GaussianDenseWithGammaPrior(units=self.latent_dim,
                                                    prior_mode="weight_and_bias")
        self.scale_head = GaussianDenseWithGammaPrior(units=self.latent_dim,
                                                      prior_mode="weight_and_bias",
                                                      activation=tf.nn.softplus)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        tensor = inputs

        for layer in self.transforms:
            tensor = layer(tensor)

        loc = self.loc_head(tensor)
        scale = self.scale_head(tensor)

        return loc, scale


class Decoder(tfl.Layer):

    def __init__(self, name="decoder", *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

    def build(self, input_shape):
        self.transforms = [
            GaussianDenseWithGammaPrior(units=512,
                                        prior_mode="weight_and_bias",
                                        activation=tf.nn.relu),
            GaussianDenseWithGammaPrior(units=1024,
                                        prior_mode="weight_and_bias",
                                        activation=tf.nn.relu),
            GaussianDenseWithGammaPrior(units=28 * 28,
                                        prior_mode="weight_and_bias",
                                        activation=tf.nn.sigmoid),
            tfl.Reshape(target_shape=(28, 28, 1)),
        ]

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        tensor = inputs

        for layer in self.transforms:
            tensor = layer(tensor)

        return tensor


class BNNLVM(tf.keras.Model):

    def __init__(self,
                 latent_dim,
                 name="bnnlvm", *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

        self.latent_dim = latent_dim

        self.encoder = Encoder(latent_dim=self.latent_dim)
        self.decoder = Decoder()

        self.prior_mean = tf.Variable(tf.zeros(latent_dim), name="prior_mean")
        self.prior_scale = tf.Variable(tf.ones(latent_dim), name="prior_scale")

        self.prior = tfd.Normal(loc=self.prior_mean, scale=self.prior_scale)

        self.likelihood_log_scale = tf.Variable(0.,
                                                name="likelihood_log_scale")

        self.log_var_prior = tfd.LogNormal(loc=tf.math.log(1e-3), scale=0.1)

    def resample_weight_prior_parameters(self):

        for layer in self.decoder.transforms:
            if isinstance(layer, GaussianDenseWithGammaPrior):
                layer.resample_precisions()

    def weight_prior_log_prob(self):

        prior_log_prob = 0.
        num_params = 0

        for layer in self.decoder.transforms:
            if isinstance(layer, GaussianDenseWithGammaPrior):
                prior_log_prob += layer.weight_log_prob()
                num_params += layer.num_params

        return prior_log_prob

    def hyper_prior_log_prob(self):

        hyperprior_log_prob = 0.
        num_params = 0

        for layer in self.decoder.transforms:
            if isinstance(layer, GaussianDenseWithGammaPrior):
                hyperprior_log_prob += layer.hyper_prior_log_prob()
                num_params += layer.num_params

        return hyperprior_log_prob  # / tf.cast(num_params, tf.float32)

    def call(self, tensor):
        posterior_loc, posterior_scale = self.encoder(tensor)

        self.posterior = tfd.Normal(loc=posterior_loc, scale=posterior_scale)

        self.kl_divergence = tfd.kl_divergence(self.posterior, self.prior)
        self.kl_divergence = tf.reduce_mean(tf.reduce_sum(self.kl_divergence, axis=1))

        latent_code = self.posterior.sample()

        reconstruction = self.decoder(latent_code)

        self.likelihood_dist = tfd.Normal(loc=reconstruction,
                                          scale=tf.exp(self.likelihood_log_scale))
        self.likelihood = self.likelihood_dist.log_prob(tensor)
        self.likelihood = tf.reduce_mean(tf.reduce_sum(self.likelihood, axis=[1, 2]))

        return reconstruction
