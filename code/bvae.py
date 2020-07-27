import tensorflow as tf
import tensorflow_probability as tfp

from conv_with_prior import GaussianConv2DWithPrior, GaussianConv2DTransposeWithPrior, ConvWithPrior
from dense_with_prior import GaussianDenseWithGammaPrior

tfl = tf.keras.layers
tfd = tfp.distributions


class Encoder(tfl.Layer):

    def __init__(self, prior_mode, latent_dim, dim_h=25, name="encoder", *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

        self.latent_dim = latent_dim
        self.dim_h = dim_h

        self.prior_mode = prior_mode

    def build(self, input_shape):
        self.transforms = [
            GaussianConv2DWithPrior(filters=16,
                                    kernel_size=(5, 5),
                                    strides=2,
                                    padding="same",
                                    activation=tf.nn.relu,
                                    prior_mode=(
                                        "per_filter" if self.prior_mode == "per_unit" else self.prior_mode),
                                    name="conv_0"),
            GaussianConv2DWithPrior(filters=32,
                                    kernel_size=(5, 5),
                                    strides=2,
                                    padding="same",
                                    activation=tf.nn.relu,
                                    prior_mode=(
                                        "per_filter" if self.prior_mode == "per_unit" else self.prior_mode),
                                    name="conv_1"),
            GaussianConv2DWithPrior(filters=32,
                                    kernel_size=(3, 3),
                                    strides=1,
                                    padding="same",
                                    activation=tf.nn.relu,
                                    prior_mode=(
                                        "per_filter" if self.prior_mode == "per_unit" else self.prior_mode),
                                    name="conv_2"),
            tfl.Reshape(target_shape=(7 * 7 * 32,)),
            GaussianDenseWithGammaPrior(units=self.dim_h,
                                        activation=tf.nn.relu,
                                        prior_mode=self.prior_mode,
                                        name="dense_0"),
        ]

        self.loc_head = GaussianDenseWithGammaPrior(units=self.latent_dim,
                                                    prior_mode=self.prior_mode,
                                                    name="dense_loc")
        self.scale_head = GaussianDenseWithGammaPrior(units=self.latent_dim,
                                                      prior_mode=self.prior_mode,
                                                      activation=tf.nn.softplus,
                                                      name="dense_scale")

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        tensor = inputs

        for layer in self.transforms:
            tensor = layer(tensor)

        loc = self.loc_head(tensor)
        scale = self.scale_head(tensor)

        return loc, scale


class Decoder(tfl.Layer):

    def __init__(self, prior_mode, name="decoder", dim_h=25, *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

        self.dim_h = dim_h
        self.prior_mode = prior_mode

    def build(self, input_shape):
        self.transforms = [
            GaussianDenseWithGammaPrior(units=self.dim_h,
                                        activation=tf.nn.relu,
                                        prior_mode=self.prior_mode,
                                        name="dense_0"),
            GaussianDenseWithGammaPrior(units=7 * 7 * 32,
                                        activation=tf.nn.relu,
                                        prior_mode=self.prior_mode,
                                        name="dense_1"),
            tfl.Reshape((7, 7, 32)),
            GaussianConv2DTransposeWithPrior(filters=32,
                                             kernel_size=(3, 3),
                                             strides=(1, 1),
                                             prior_mode=(
                                                 "per_filter" if self.prior_mode == "per_unit" else self.prior_mode),
                                             padding="same",
                                             name="deconv_0"),
            GaussianConv2DTransposeWithPrior(filters=16,
                                             kernel_size=(5, 5),
                                             strides=2,
                                             padding="same",
                                             prior_mode=(
                                                 "per_filter" if self.prior_mode == "per_unit" else self.prior_mode),
                                             name="deconv_1"),
            GaussianConv2DTransposeWithPrior(filters=1,
                                             kernel_size=(5, 5),
                                             strides=2,
                                             padding="same",
                                             prior_mode=(
                                                 "per_filter" if self.prior_mode == "per_unit" else self.prior_mode),
                                             name="deconv_2"),
        ]

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        tensor = inputs

        for layer in self.transforms:
            tensor = layer(tensor)

        return tensor


class BVAE(tf.keras.Model):

    def __init__(self, latent_dim, prior_mode="weigth_and_bias", name="bvae", *args, **kwargs):

        super().__init__(name=name, *args, **kwargs)

        self.latent_dim = latent_dim
        self.prior_mode = prior_mode

        self.encoder = Encoder(latent_dim=self.latent_dim,
                               prior_mode=prior_mode)

        self.decoder = Decoder(prior_mode=self.prior_mode)

        self.prior_mean = tf.Variable(tf.zeros(latent_dim), name="prior_mean")
        self.prior_scale = tf.Variable(tf.ones(latent_dim), name="prior_scale")

        self.prior = tfd.Normal(loc=self.prior_mean, scale=self.prior_scale)

    def resample_weight_prior_parameters(self, kind):

        if kind == "encoder":
            transforms = self.encoder.transforms
        elif kind == "decoder":
            transforms = self.decoder.transforms
        else:
            raise NotImplementedError

        for layer in transforms:
            if isinstance(layer, (GaussianDenseWithGammaPrior,
                                  ConvWithPrior)):
                layer.resample_precisions()

    def weight_prior_log_prob(self, kind):

        if kind == "encoder":
            transforms = self.encoder.transforms
        elif kind == "decoder":
            transforms = self.decoder.transforms
        else:
            raise NotImplementedError

        prior_log_prob = 0.
        num_params = 0

        for layer in transforms:
            if isinstance(layer, (GaussianDenseWithGammaPrior, GaussianConv2DWithPrior)):
                prior_log_prob += layer.weight_log_prob()
                num_params += layer.num_params

        return prior_log_prob

    def hyper_prior_log_prob(self, kind):

        if kind == "encoder":
            transforms = self.encoder.transforms
        elif kind == "decoder":
            transforms = self.decoder.transforms
        else:
            raise NotImplementedError

        hyperprior_log_prob = 0.
        num_params = 0

        for layer in transforms:
            if isinstance(layer, (GaussianDenseWithGammaPrior, GaussianConv2DWithPrior)):
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

        self.likelihood_dist = tfd.Normal(loc=reconstruction, scale=1.)
        self.likelihood = self.likelihood_dist.log_prob(tensor)
        self.likelihood = tf.reduce_mean(tf.reduce_sum(self.likelihood, axis=[1, 2, 3]))

        return reconstruction
