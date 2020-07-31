import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_datasets as tfds

import datetime

from sacred import Experiment

from vae import VAE
from bvae import BVAE

from sghmc import SGHMC
from adaptive_sghmc_v2 import AdaptiveSGHMC

from dense_with_prior import GaussianDenseWithGammaPrior
from conv_with_prior import GaussianConv2DWithPrior, GaussianConv2DTransposeWithPrior

tfs = tf.summary
tfd = tfp.distributions

ex = Experiment("bvae_experiment", ingredients=[])


@ex.config
def config():

    data_dir = "/scratch/gf332/Misc/datasets/"
    model_base_save_dir = "/scratch/gf332/Misc/bvae_experiments"

    # Model options: vae, bvae-decoder, bvae-full
    model_type = "vae"
    likelihood = "gaussian"
    prior_mode = "weight_and_bias"

    dataset_name = "fashion_mnist"
    dataset_size = 60000
    batch_size = 500

    burnin = 50

    iterations = 100000
    learning_rate = 1e-3

    # Latent dimensions of the VAE
    latent_dim = 20

    # Logging
    tensorboard_log_freq = 1000

    model_save_dir = f"{model_base_save_dir}/{dataset_name}/{model_type}/{likelihood}_likelihood/latent_dim_{latent_dim}"

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"{model_save_dir}/logs/{current_time}/train"


@ex.automain
def train(model_save_dir,
          data_dir,
          model_type,
          likelihood,
          prior_mode,
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

    # -------------------------------------------------------------------------
    # Prepare the dataset
    # -------------------------------------------------------------------------
    num_batch_per_epoch = dataset_size // batch_size
    _log.info(f"{num_batch_per_epoch} batches per epoch!")

    data = tfds.load(dataset_name, data_dir=data_dir)

    train_data = data["train"]
    train_data = train_data.map(lambda x: tf.cast(x["image"], tf.float32) / 255.)
    train_data = train_data.shuffle(5000)
    train_data = train_data.repeat()
    train_data = train_data.batch(batch_size)
    train_data = train_data.prefetch(16)

    # -------------------------------------------------------------------------
    # Create model
    # -------------------------------------------------------------------------

    model = BVAE(latent_dim=latent_dim,
                 likelihood=likelihood,
                 prior_mode=prior_mode)

    model.build(input_shape=(batch_size, 28, 28, 1))

    learning_rate = tf.Variable(learning_rate, dtype=tf.float32, name="learn_rate")

    encoder_optimizer = {
        "vae": tf.optimizers.Adam(learning_rate=1e-3),
        "bvae-decoder": tf.optimizers.Adam(learning_rate=1e-3),
        "bvae-full": AdaptiveSGHMC(learning_rate=learning_rate,
                                   burnin=num_batch_per_epoch * burnin,
                                   data_size=dataset_size,
                                   overestimation_rate=1,
                                   initialization_rounds=10,
                                   friction=0.05),
    }[model_type]

    decoder_optimizer = {
        "vae": tf.optimizers.Adam(learning_rate=1e-3),

        "bvae-decoder": AdaptiveSGHMC(learning_rate=learning_rate,
                                   burnin=num_batch_per_epoch * burnin,
                                   data_size=dataset_size,
                                   overestimation_rate=1,
                                   initialization_rounds=10,
                                   friction=0.05),

        "bvae-full": AdaptiveSGHMC(learning_rate=learning_rate,
                                   burnin=num_batch_per_epoch * burnin,
                                   data_size=dataset_size,
                                   overestimation_rate=1,
                                   initialization_rounds=10,
                                   friction=0.05),
    }[model_type]

    ckpt = tf.train.Checkpoint(step=tf.Variable(1, dtype=tf.int64),
                               learning_rate=learning_rate,
                               model=model,
                               encoder_optimizer=encoder_optimizer,
                               decoder_optimizer=decoder_optimizer)

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

            if model_type == "bvae-full":
                model.resample_weight_prior_parameters(kind="encoder")
                tf.print("Resampling encoder weight scales!")

            if model_type in ["bvae-decoder", "bvae-full"]:
                model.resample_weight_prior_parameters(kind="decoder")
                tf.print("Resampling decoder weight scales!")

        weight_log_probs = []
        hyperprior_log_probs = []

        with tf.GradientTape(persistent=True) as tape:

            data_size = tf.cast(dataset_size, tf.float32)

            reconstructions = model(batch)

            elbo = model.data_log_likelihood - beta * model.kl_divergence

            encoder_loss = -elbo
            decoder_loss = -elbo

            if model_type == "bvae-full":
                encoder_wlp = model.weight_prior_log_prob(kind="encoder") / data_size
                encoder_hplp = model.hyper_prior_log_prob(kind="encoder") / data_size

                weight_log_probs.append(("encoder", encoder_wlp))
                hyperprior_log_probs.append(("encoder", encoder_hplp))

                encoder_loss += -(encoder_wlp + encoder_hplp)

            if model_type in ["bvae-decoder", "bvae-full"]:
                decoder_wlp = model.weight_prior_log_prob(kind="decoder") / data_size
                decoder_hplp = model.hyper_prior_log_prob(kind="decoder") / data_size

                weight_log_probs.append(("decoder", decoder_wlp))
                hyperprior_log_probs.append(("decoder", decoder_hplp))

                decoder_loss += -(decoder_wlp + decoder_hplp)

        encoder_gradients = tape.gradient(encoder_loss, model.encoder.trainable_variables)
        encoder_optimizer.apply_gradients(zip(encoder_gradients, model.encoder.trainable_variables))

        extra_decoder_vars = []

        if model.likelihood == "gaussian":
            extra_decoder_vars = [model.likelihood_log_scale]

        decoder_gradients = tape.gradient(decoder_loss, model.decoder.trainable_variables + extra_decoder_vars)
        decoder_optimizer.apply_gradients(zip(decoder_gradients, model.decoder.trainable_variables + extra_decoder_vars))

        del tape

        individual_kls = tf.reduce_mean(tfd.kl_divergence(model.posterior, model.prior), axis=0)

        return reconstructions, model.data_log_likelihood, model.kl_divergence, weight_log_probs, hyperprior_log_probs, individual_kls

    for batch in train_data:

        ckpt.step.assign_add(1)

        beta = tf.minimum((1. / tf.cast(anneal_epochs * num_batch_per_epoch, tf.float32)) * tf.cast(ckpt.step, tf.float32), 1.)

        reconstructions, log_likelihood, kl_divergence, prior_log_probs, hyperprior_log_probs, individual_kls = train_step(model, batch, beta)

        if int(ckpt.step) % num_batch_per_epoch == 0:

            # Save model
            save_path = manager.save()
            _log.info(f"Step {int(ckpt.step)}: Saved model to {save_path}")

            with summary_writer.as_default():
                tfs.scalar(name="Likelihood", data=log_likelihood, step=ckpt.step)
                tfs.scalar(name="Total_KL", data=kl_divergence, step=ckpt.step)
                tfs.scalar(name="ELBO", data=log_likelihood - kl_divergence, step=ckpt.step)
                tfs.scalar(name="Beta", data=beta, step=ckpt.step)

                for name, prior_log_prob in prior_log_probs:
                    tfs.scalar(name=f"Prior_log_prob/{name}", data=prior_log_prob, step=ckpt.step)

                for name, hyperprior_log_prob in hyperprior_log_probs:
                    tfs.scalar(name=f"Hyperprior_log_prob/{name}", data=hyperprior_log_prob, step=ckpt.step)

                tfs.image(name="Original", data=batch, step=ckpt.step)
                tfs.image(name="Reconstruction", data=reconstructions, step=ckpt.step)

                if model.likelihood == "gaussian":
                    tfs.scalar(name="Likelihood_scale", data=tf.exp(model.likelihood_log_scale), step=ckpt.step)

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



