import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfl = tf.keras.layers


class Encoder(tfl.Layer):

    def __init__(self, latent_dim, name="encoder", *args, **kwargs):

        super().__init__(name=name, *args, **kwargs)

        self.latent_dim = latent_dim

    def build(self, input_shape):

        self.transforms = [
            tfl.Reshape(target_shape=(28 * 28,)),
            tfl.Dense(units=1024,
                      activation=tf.nn.relu),
            tfl.Dense(units=512,
                      activation=tf.nn.relu),
        ]

        self.loc_head = tfl.Dense(units=self.latent_dim)
        self.scale_head = tfl.Dense(units=self.latent_dim,
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
            tfl.Dense(units=512,
                      activation=tf.nn.relu),
            tfl.Dense(units=1024,
                      activation=tf.nn.relu),
            tfl.Dense(units=28 * 28,
                      activation=tf.nn.sigmoid),
            tfl.Reshape(target_shape=(28, 28, 1)),
        ]

        super().build(input_shape)

    def call(self, inputs, **kwargs):

        tensor = inputs

        for layer in self.transforms:
            tensor = layer(tensor)

        return tensor


class VAE(tf.keras.Model):

    def __init__(self, latent_dim, name="vae", *args, **kwargs):

        super().__init__(name=name, *args, **kwargs)

        self.latent_dim = latent_dim

        self.encoder = Encoder(latent_dim=self.latent_dim)
        self.decoder = Decoder()

        self.prior_mean = tf.Variable(tf.zeros(latent_dim), name="prior_mean")
        self.prior_scale = tf.Variable(tf.ones(latent_dim), name="prior_scale")

        self.prior = tfd.Normal(loc=self.prior_mean, scale=self.prior_scale)

    def call(self, tensor):

        posterior_loc, posterior_scale = self.encoder(tensor)

        self.posterior = tfd.Normal(loc=posterior_loc, scale=posterior_scale)

        self.kl_divergence = tfd.kl_divergence(self.posterior, self.prior)
        self.kl_divergence = tf.reduce_mean(tf.reduce_sum(self.kl_divergence, axis=1))

        latent_code = self.posterior.sample()

        reconstruction = self.decoder(latent_code)

        self.likelihood_dist = tfd.Normal(loc=reconstruction, scale=1.)
        self.likelihood = self.likelihood_dist.log_prob(tensor)
        self.likelihood = tf.reduce_mean(tf.reduce_sum(self.likelihood, axis=[1, 2]))

        return reconstruction