import abc

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


class ConvWithPrior(abc.ABC):
    _AVAILABLE_PRIOR_MODES = [
        "per_param",
        "weight_and_bias",
        "per_filter"
    ]

    def __init__(self,
                 prior_mode,
                 alpha0=1.,
                 beta0=1.,
                 eps=1e-6):

        assert isinstance(self, tf.keras.layers.Conv2D)

        if prior_mode not in self._AVAILABLE_PRIOR_MODES:
            raise ValueError(f"Prior mode must be one of {self._AVAILABLE_PRIOR_MODES}, but '{prior_mode}' was given!")

        self.prior_mode = prior_mode
        self.alpha0 = alpha0
        self.beta0 = beta0

        self.eps = eps
        self.eps_squared = self.eps ** 2.

    @property
    def num_params(self):
        return tf.size(self.kernel) + (tf.size(self.bias) if self.use_bias else 0)

    def weight_log_prob(self):
        log_prob = tf.reduce_sum(self.kernel_prior.log_prob(self.kernel))

        if self.use_bias:
            log_prob += tf.reduce_sum(self.bias_prior.log_prob(self.bias))

        return log_prob

    def hyper_prior_log_prob(self):

        log_prob = tf.reduce_sum(self.kernel_prec_hyper_prior.log_prob(self.scale_to_prec(self.kernel_scale)))

        if self.use_bias:
            if self.prior_mode == "per_filter":
                bias_hyperprior = self.kernel_prec_hyper_prior
            else:
                bias_hyperprior = self.bias_prec_hyper_prior

            log_prob += tf.reduce_sum(bias_hyperprior.log_prob(self.scale_to_prec(self.bias_scale)))

        return log_prob

    def scale_to_prec(self, scale):
        return 1. / (tf.square(scale) + self.eps)

    def prec_to_scale(self, prec):
        return 1. / tf.sqrt(prec + self.eps_squared)

    def _resample_precisions(self, per_filter_rate_sum_axes):


        per_filter_kernel_scale_reshape = [(1 if i in per_filter_rate_sum_axes else self.filters)
                                           for i in range(4)]

        # Perform Gibbs step for appropriate prior mode
        if self.prior_mode == "per_param":

            kernel_conc = self.alpha0 + 0.5
            kernel_conc = tf.ones_like(self.kernel) * kernel_conc
            kernel_rate = self.beta0 + tf.square(self.kernel) / 2.

            if self.use_bias:
                bias_conc = self.alpha0 + 0.5
                bias_conc = tf.ones_like(self.bias) * bias_conc
                bias_rate = self.beta0 + tf.square(self.bias) / 2.

        elif self.prior_mode == "weight_and_bias":

            kernel_conc = self.alpha0 + tf.cast(tf.size(self.kernel), self.kernel.dtype) / 2.
            kernel_rate = self.beta0 + tf.reduce_sum(tf.square(self.kernel)) / 2.

            if self.use_bias:
                bias_conc = self.alpha0 + tf.cast(tf.size(self.bias), self.bias.dtype) / 2.
                bias_rate = self.beta0 + tf.reduce_sum(tf.square(self.bias)) / 2.

        elif self.prior_mode == "per_filter":

            n_elems_per_filter = tf.reduce_prod(self.kernel.shape[:-1])

            conc = self.alpha0 + tf.cast(n_elems_per_filter, self.kernel.dtype) / 2.
            rate = self.beta0 + tf.reduce_sum(tf.square(self.kernel), axis=per_filter_rate_sum_axes) / 2.

            if self.use_bias:
                conc = conc + 0.5
                rate = rate + tf.square(self.bias) / 2.

                bias_conc = tf.ones_like(self.bias) * conc
                bias_rate = rate

            kernel_conc = tf.ones(self.filters) * conc
            kernel_rate = rate

        else:
            raise NotImplementedError

        self.kernel_conc.assign(kernel_conc)
        self.kernel_rate.assign(kernel_rate)

        if self.use_bias and self.prior_mode != "per_filter":
            self.bias_conc.assign(bias_conc)
            self.bias_rate.assign(bias_rate)

        # Sample from the posteriors
        new_kernel_scale = self.prec_to_scale(self.kernel_prec_hyper_prior.sample())

        if self.prior_mode == "per_param":
            assigned_new_kernel_scale = new_kernel_scale
        elif self.prior_mode == "per_filter":
            assigned_new_kernel_scale = tf.ones_like(self.kernel) * tf.reshape(new_kernel_scale, per_filter_kernel_scale_reshape)
        elif self.prior_mode == "weight_and_bias":
            assigned_new_kernel_scale = tf.ones_like(self.kernel) * new_kernel_scale
        else:
            raise NotImplementedError

        self.kernel_scale.assign(assigned_new_kernel_scale)

        if self.use_bias:

            if self.prior_mode != "per_filter":
                new_bias_scale = self.prec_to_scale(self.bias_prec_hyper_prior.sample())

            if self.prior_mode == "per_param":
                pass
            elif self.prior_mode == "per_filter":
                new_bias_scale = new_kernel_scale
            elif self.prior_mode == "weight_and_bias":
                new_bias_scale = tf.ones(self.filters) * new_bias_scale
            else:
                raise NotImplementedError

            self.bias_scale.assign(new_bias_scale)

    def build_priors(self, scale_init=0.1):
        kernel_hyperprior_shape = {
            "per_param": self.kernel.shape,
            "per_filter": (self.filters,),
            "weight_and_bias": (),
        }[self.prior_mode]

        self.kernel_conc = self.add_weight(
            'kernel_conc',
            shape=kernel_hyperprior_shape,
            initializer=tf.constant_initializer(value=self.alpha0),
            regularizer=None,
            constraint=None,
            dtype=self.dtype,
            trainable=False)

        self.kernel_rate = self.add_weight(
            'kernel_rate',
            shape=kernel_hyperprior_shape,
            initializer=tf.constant_initializer(value=self.beta0),
            regularizer=None,
            constraint=None,
            dtype=self.dtype,
            trainable=False)

        self.kernel_prec_hyper_prior = tfd.Gamma(concentration=self.kernel_conc,
                                                 rate=self.kernel_rate)

        self.kernel_scale = self.add_weight(
            'kernel_scale',
            shape=self.kernel.shape,
            initializer=tf.constant_initializer(value=scale_init),
            regularizer=None,
            constraint=None,
            dtype=self.dtype,
            trainable=False)

        self.kernel_prior = tfd.Normal(loc=tf.zeros_like(self.kernel),
                                       scale=self.kernel_scale)

        if self.use_bias:

            self.bias_scale = self.add_weight(
                'bias_scale',
                shape=[self.filters, ],
                initializer=tf.constant_initializer(value=scale_init),
                regularizer=None,
                constraint=None,
                dtype=self.dtype,
                trainable=False)

            if self.prior_mode != "per_filter":
                bias_hyperprior_shape = {
                    "per_param": (self.filters,),
                    "weight_and_bias": (),
                }[self.prior_mode]

                self.bias_conc = self.add_weight(
                    'bias_conc',
                    shape=bias_hyperprior_shape,
                    initializer=tf.constant_initializer(value=self.alpha0),
                    regularizer=None,
                    constraint=None,
                    dtype=self.dtype,
                    trainable=False)

                self.bias_rate = self.add_weight(
                    'bias_rate',
                    shape=bias_hyperprior_shape,
                    initializer=tf.constant_initializer(value=self.beta0),
                    regularizer=None,
                    constraint=None,
                    dtype=self.dtype,
                    trainable=False)

                self.bias_prec_hyper_prior = tfd.Gamma(concentration=self.bias_conc,
                                                       rate=self.bias_rate)

            self.bias_prior = tfd.Normal(loc=tf.zeros_like(self.bias),
                                         scale=self.bias_scale)


class GaussianConv2DWithPrior(tf.keras.layers.Conv2D, ConvWithPrior):

    def __init__(self,
                 filters,
                 kernel_size,
                 prior_mode,
                 alpha0=1.,
                 beta0=1.,
                 strides=(1, 1),
                 padding='valid',
                 activation=None,
                 use_bias=True,
                 name="gaussian_conv2d",
                 **kwargs,
                 ):
        tf.keras.layers.Conv2D.__init__(self,
                                        filters=filters,
                                        kernel_size=kernel_size,
                                        strides=strides,
                                        padding=padding,
                                        activation=activation,
                                        use_bias=use_bias,
                                        name=name,
                                        **kwargs)

        ConvWithPrior.__init__(self,
                               prior_mode=prior_mode,
                               alpha0=alpha0,
                               beta0=beta0)

    def resample_precisions(self):

        self._resample_precisions(per_filter_rate_sum_axes=[0, 1, 2])

    def build(self, input_shape, scale_init=0.1):
        super().build(input_shape)
        super().build_priors(scale_init=scale_init)


class GaussianConv2DTransposeWithPrior(tf.keras.layers.Conv2DTranspose, ConvWithPrior):

    def __init__(self,
                 filters,
                 kernel_size,
                 prior_mode,
                 alpha0=1.,
                 beta0=1.,
                 strides=(1, 1),
                 padding='valid',
                 activation=None,
                 use_bias=True,
                 name="gaussian_conv2d",
                 **kwargs,
                 ):

        tf.keras.layers.Conv2DTranspose.__init__(self,
                                                 filters=filters,
                                                 kernel_size=kernel_size,
                                                 strides=strides,
                                                 padding=padding,
                                                 activation=activation,
                                                 use_bias=use_bias,
                                                 name=name,
                                                 **kwargs)

        ConvWithPrior.__init__(self,
                               prior_mode=prior_mode,
                               alpha0=alpha0,
                               beta0=beta0)

    def resample_precisions(self):
        self._resample_precisions(per_filter_rate_sum_axes=[0, 1, 3])

    def build(self, input_shape, scale_init=0.1):

        super().build(input_shape)
        super().build_priors(scale_init=scale_init)
