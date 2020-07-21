import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_datasets as tfds

import datetime

from sacred import Experiment

from vae import VAE
from bvae import BVAE

from sghmc import SGHMC
from adaptive_sghmc import AdaptiveSGHMC

from dense_with_prior import GaussianDenseWithGammaPrior
from conv_with_prior import GaussianConv2DWithPrior, GaussianConv2DTransposeWithPrior

tfs = tf.summary
tfd = tfp.distributions

ex = Experiment("bvae_experiment", ingredients=[])


@ex.config
def config():

    data_dir = "/scratch/gf332/Misc/datasets/"
    model_base_save_dir = "/scratch/gf332/Misc/bvae_experiments"

    model = "bvae"

    dataset_size = 60000
    batch_size = 500

    # Gradient descent Optimizer
    optimizer = "adam"
    iterations = 100000
    learning_rate = 1e-3

    # Latent dimensions of the VAE
    latent_dim = 20

    # Logging
    tensorboard_log_freq = 1000

    model_save_dir = f"{model_base_save_dir}/{model}/latent_dim_{latent_dim}"

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"{model_save_dir}/logs/{current_time}/train"


@ex.automain
def train(model_save_dir,
          data_dir,
          model,
          dataset_size,

          batch_size,

          optimizer,
          iterations,
          learning_rate,

          latent_dim,
          log_dir,
          tensorboard_log_freq,

          _log,
          ):

    # -------------------------------------------------------------------------
    # Prepare the dataset
    # -------------------------------------------------------------------------
    num_batch_per_epoch = dataset_size // batch_size
    _log.info(f"{num_batch_per_epoch} batches per epoch!")

    data = tfds.load("mnist", data_dir=data_dir)

    train_data = data["train"]
    train_data = train_data.map(lambda x: tf.cast(x["image"], tf.float32) / 255.)
    train_data = train_data.shuffle(5000)
    train_data = train_data.repeat()
    train_data = train_data.batch(batch_size)
    train_data = train_data.prefetch(16)

    # -------------------------------------------------------------------------
    # Create model
    # -------------------------------------------------------------------------

    model = {
        "vae": VAE(latent_dim=latent_dim),
        "bvae": BVAE(latent_dim=latent_dim,
                     prior_mode="weight_and_bias")
    }[model]

    model.build(input_shape=(batch_size, 28, 28, 1))

    learning_rate = tf.Variable(learning_rate, dtype=tf.float32, name="learn_rate")

    optimizer = {
        "adam": tf.optimizers.Adam
    }[optimizer](learning_rate=1e-3)

    sampler = {
        "sghmc": SGHMC(learning_rate=3e-8,
                       data_size=dataset_size,
                       momentum_decay=0.05),

        "adaptive_sghmc": AdaptiveSGHMC(learning_rate=1e-3,
                                        burnin=num_batch_per_epoch * 50,
                                        data_size=dataset_size,
                                        overestimation_rate=1,
                                        initialization_rounds=10,
                                        momentum_decay=0.05),
    }["adaptive_sghmc"]

    ckpt = tf.train.Checkpoint(step=tf.Variable(1, dtype=tf.int64),
                               learning_rate=learning_rate,
                               model=model,
                               optimizer=optimizer,
                               sampler=sampler)

    manager = tf.train.CheckpointManager(ckpt, model_save_dir, max_to_keep=3)

    # -------------------------------------------------------------------------
    # Create Summary Writer
    # -------------------------------------------------------------------------
    summary_writer = tfs.create_file_writer(log_dir)

    # -------------------------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------------------------

    anneal_epochs = 100

    # Restore previous session
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        _log.info(f"Restored model from {manager.latest_checkpoint}")
    else:
        _log.info("Initializing model from scratch.")

    @tf.function
    def train_step(model, batch, beta):

        if int(ckpt.step) % num_batch_per_epoch == 0:
            tf.print("Resampling weight scales!")
            model.resample_weight_prior_parameters()

        with tf.GradientTape(persistent=True) as tape:

            reconstructions = model(batch)

            elbo = model.likelihood - beta * model.kl_divergence

            prior_log_prob = model.weight_prior_log_prob() / tf.cast(dataset_size, tf.float32)

            hyperprior_log_prob = model.hyper_prior_log_prob() / tf.cast(dataset_size, tf.float32)

            encoder_loss = -elbo
            decoder_loss = -(elbo + prior_log_prob + hyperprior_log_prob)

        encoder_gradients = tape.gradient(encoder_loss, model.encoder.trainable_variables)
        optimizer.apply_gradients(zip(encoder_gradients, model.encoder.trainable_variables))

        decoder_gradients = tape.gradient(decoder_loss, model.decoder.trainable_variables)
        sampler.apply_gradients(zip(decoder_gradients, model.decoder.trainable_variables))

        del tape

        individual_kls = tf.reduce_mean(tfd.kl_divergence(model.posterior, model.prior), axis=0)

        return reconstructions, model.likelihood, model.kl_divergence, prior_log_prob, hyperprior_log_prob, individual_kls

    for batch in train_data:

        ckpt.step.assign_add(1)

        beta = tf.minimum((1. / tf.cast(anneal_epochs * num_batch_per_epoch, tf.float32)) * tf.cast(ckpt.step, tf.float32), 1.)

        reconstructions, likelihood, kl_divergence, prior_log_prob, hyperprior_log_prob, individual_kls = train_step(model, batch, beta)

        if int(ckpt.step) % num_batch_per_epoch == 0:

            # Save model
            save_path = manager.save()
            _log.info(f"Step {int(ckpt.step)}: Saved model to {save_path}")


            with summary_writer.as_default():
                tfs.scalar(name="Likelihood", data=likelihood, step=ckpt.step)
                tfs.scalar(name="Total_KL", data=kl_divergence, step=ckpt.step)
                tfs.scalar(name="ELBO", data=likelihood - kl_divergence, step=ckpt.step)
                tfs.scalar(name="Beta", data=beta, step=ckpt.step)

                tfs.scalar(name="Prior_log_prob", data=prior_log_prob, step=ckpt.step)
                tfs.scalar(name="Hyperprior_log_prob", data=hyperprior_log_prob, step=ckpt.step)

                tfs.image(name="Original", data=batch, step=ckpt.step)
                tfs.image(name="Reconstruction", data=reconstructions, step=ckpt.step)

                for i, kl in enumerate(individual_kls):
                    tfs.scalar(f"KL/dim_{i}", data=kl, step=ckpt.step)

                for layer in model.decoder.transforms:
                    if isinstance(layer, (GaussianDenseWithGammaPrior, GaussianConv2DWithPrior, GaussianConv2DTransposeWithPrior)):
                        tfs.scalar(name=f"rate/{layer.name}", data=tf.reduce_mean(layer.kernel_rate), step=ckpt.step)
                        tfs.scalar(name=f"avg_weight/{layer.name}", data=tf.reduce_mean(layer.kernel), step=ckpt.step)
                        tfs.scalar(name=f"max_weight/{layer.name}", data=tf.reduce_max(layer.kernel), step=ckpt.step)
                        tfs.scalar(name=f"min_weight/{layer.name}", data=tf.reduce_min(layer.kernel), step=ckpt.step)

                    if isinstance(layer, GaussianDenseWithGammaPrior):
                        tfs.scalar(name=f"weight_00/{layer.name}", data=layer.kernel[0, 0], step=ckpt.step)
                        tfs.scalar(name=f"scale_00/{layer.name}", data=layer.kernel_scale[0, 0], step=ckpt.step)
                        tfs.scalar(name=f"prec_00/{layer.name}",
                                   data=layer.scale_to_prec(layer.kernel_scale[0, 0]),
                                   step=ckpt.step)



