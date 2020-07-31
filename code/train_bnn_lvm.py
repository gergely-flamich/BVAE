import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_datasets as tfds

import datetime

from sacred import Experiment

from adaptive_sghmc_v2 import AdaptiveSGHMC

from dense_with_prior import GaussianDenseWithGammaPrior
from conv_with_prior import GaussianConv2DWithPrior, GaussianConv2DTransposeWithPrior

from bnn_lvm import BNNLVM

tfs = tf.summary
tfd = tfp.distributions

ex = Experiment("bvae_experiment", ingredients=[])


@ex.config
def config():
    data_dir = "/scratch/gf332/Misc/datasets/"
    model_base_save_dir = "/scratch/gf332/Misc/bnn_lvm_experiments"

    # Model options: vae, bvae-encoder, bvae-full
    prior_mode = "weight_and_bias"

    dataset_name = "mnist"
    dataset_size = 60000
    batch_size = 500

    burnin = 20

    iterations = 100000
    learning_rate = 1e-3

    # Latent dimensions of the VAE
    latent_dim = 2

    # Logging
    tensorboard_log_freq = 1000

    model_save_dir = f"{model_base_save_dir}/{dataset_name}/latent_dim_{latent_dim}"

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"{model_save_dir}/logs/{current_time}/train"


@ex.automain
def train(model_save_dir,
          data_dir,
          dataset_name,
          dataset_size,

          burnin,

          batch_size,

          learning_rate,

          latent_dim,
          log_dir,
          tensorboard_log_freq,

          _log,
          ):

    data_size = 5000
    # -------------------------------------------------------------------------
    # Prepare the dataset
    # -------------------------------------------------------------------------
    num_batch_per_epoch = dataset_size // batch_size
    _log.info(f"{num_batch_per_epoch} batches per epoch!")

    data = tfds.load(dataset_name, data_dir=data_dir)

    train_data = data["train"]
    train_data = train_data.map(lambda x: tf.cast(x["image"], tf.float32) / 255.)

    # TODO: Change later
    train_data = train_data.batch(data_size).take(1)

    for td in train_data:
        train_data_tensor = td

    # -------------------------------------------------------------------------
    # Create model
    # -------------------------------------------------------------------------

    latent_prior = tfd.Normal(loc=tf.zeros(latent_dim), scale=tf.ones(latent_dim))

    latents = tf.Variable(latent_prior.sample(train_data_tensor.shape[0]),
                          name="latents")

    model = BNNLVM(latent_dim=latent_dim)

    model.build(input_shape=(data_size, 28, 28, 1))

    optimizer = AdaptiveSGHMC(learning_rate=1e-3,
                              burnin=num_batch_per_epoch * burnin,
                              data_size=data_size,
                              overestimation_rate=1,
                              initialization_rounds=10,
                              friction=0.05)

    ckpt = tf.train.Checkpoint(step=tf.Variable(1, dtype=tf.int64),
                               model=model,
                               latents=latents,
                               optimizer=optimizer)

    manager = tf.train.CheckpointManager(ckpt, model_save_dir, max_to_keep=3)

    # -------------------------------------------------------------------------
    # Create Summary Writer
    # -------------------------------------------------------------------------
    summary_writer = tfs.create_file_writer(log_dir)

    # -------------------------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------------------------

    # Restore previous session
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        _log.info(f"Restored model from {manager.latest_checkpoint}")
    else:
        _log.info("Initializing model from scratch.")

    for i in range(300000):

        ckpt.step.assign_add(1)

        if int(ckpt.step) % 500 == 0:
            model.resample_weight_prior_parameters()
            print("Resampled hypers!")

        with tf.GradientTape() as tape:

            reconstructions = model.decoder(latents)

            lvp = tfd.LogNormal(loc=tf.math.log(1e-4), scale=0.1)
            likelihood_dist = tfd.Normal(loc=reconstructions,
                                         scale=tf.exp(model.likelihood_log_scale))

            ll = likelihood_dist.log_prob(train_data_tensor)
            ll = tf.reduce_mean(tf.reduce_sum(ll, axis=[1, 2, 3]))

            log_var_prior_lp = lvp.log_prob(tf.exp(2. * model.likelihood_log_scale))

            prior_lp = model.prior.log_prob(latents)
            prior_lp = tf.reduce_mean(tf.reduce_sum(prior_lp, axis=1))

            weight_prior_lp = model.weight_prior_log_prob() / tf.cast(data_size, tf.float32)
            weight_hyperprior_lp = model.hyper_prior_log_prob() / tf.cast(data_size, tf.float32)

            loss = -(ll + prior_lp + weight_prior_lp + weight_hyperprior_lp + log_var_prior_lp)

        gradients = tape.gradient(loss, [latents, model.likelihood_log_scale] + model.decoder.trainable_variables)
        optimizer.apply_gradients(zip(gradients, [latents, model.likelihood_log_scale] + model.decoder.trainable_variables))

        if int(ckpt.step) % 100 == 0:

            # Save model
            save_path = manager.save()
            _log.info(f"Step {int(ckpt.step)}: Saved model to {save_path}")

            with summary_writer.as_default():
                tfs.scalar(name="Loss", data=loss, step=ckpt.step)
                tfs.scalar(name="Log-Likelihood", data=ll, step=ckpt.step)
                tfs.scalar(name="Latent_Prior-LP", data=prior_lp, step=ckpt.step)
                tfs.scalar(name="Weight_Prior-LP", data=weight_prior_lp, step=ckpt.step)
                tfs.scalar(name="Weight_HyperPrior-LP", data=weight_hyperprior_lp, step=ckpt.step)
                tfs.scalar(name="Log_var-LP", data=log_var_prior_lp, step=ckpt.step)
                tfs.scalar(name="likelihood_scale", data=tf.exp(model.likelihood_log_scale), step=ckpt.step)

                tfs.image(name="Original", data=train_data_tensor, step=ckpt.step)
                tfs.image(name="Reconstruction", data=reconstructions, step=ckpt.step)

