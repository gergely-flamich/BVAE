import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_datasets as tfds

import datetime

from sacred import Experiment

from vae import VAE

tfs = tf.summary
tfd = tfp.distributions

ex = Experiment("bvae_experiment", ingredients=[])


@ex.config
def config():

    data_dir = "/scratch/gf332/Misc/datasets/"
    model_base_save_dir = "/scratch/gf332/Misc/bvae_experiments"

    model = "vae"

    batch_size = 64

    # Gradient descent Optimizer
    optimizer = "adam"
    iterations = 100000
    learning_rate = 1e-3

    # Latent dimensions of the VAE
    latent_dim = 10

    # Logging
    tensorboard_log_freq = 1000

    model_save_dir = f"{model_base_save_dir}/{model}/latent_dim_{latent_dim}"

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"{model_save_dir}/logs/{current_time}/train"


@ex.automain
def train(model_save_dir,
          data_dir,
          model,

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

    model = VAE(latent_dim=latent_dim)
    model.build(input_shape=(batch_size, 28, 28, 1))

    learning_rate = tf.Variable(learning_rate, dtype=tf.float32, name="learn_rate")

    optimizer = {
        "adam": tf.optimizers.Adam
    }[optimizer](learning_rate=learning_rate)

    ckpt = tf.train.Checkpoint(step=tf.Variable(1, dtype=tf.int64),
                               learning_rate=learning_rate,
                               model=model,
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

    def train_step(model, batch):

        with tf.GradientTape() as tape:

            reconstructions = model(batch)

            elbo = model.likelihood - model.kl_divergence

            loss = -elbo

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        return reconstructions, model.likelihood, model.kl_divergence

    for batch in train_data:

        ckpt.step.assign_add(1)

        reconstructions, likelihood, kl_divergence = train_step(model, batch)

        if int(ckpt.step) % tensorboard_log_freq == 0:
            # Save model
            save_path = manager.save()
            _log.info(f"Step {int(ckpt.step)}: Saved model to {save_path}")

            individual_kls = tf.reduce_mean(tfd.kl_divergence(model.posterior, model.prior), axis=0)

            with summary_writer.as_default():
                tfs.scalar(name="Likelihood", data=likelihood, step=ckpt.step)
                tfs.scalar(name="Total_KL", data=kl_divergence, step=ckpt.step)
                tfs.scalar(name="ELBO", data=likelihood + kl_divergence, step=ckpt.step)

                tfs.image(name="Original", data=batch, step=ckpt.step)
                tfs.image(name="Reconstruction", data=reconstructions, step=ckpt.step)

                for i, kl in enumerate(individual_kls):
                    tfs.scalar(f"KL/dim_{i}", data=kl, step=ckpt.step)



