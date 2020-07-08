import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_datasets as tfds

import datetime

from sacred import Experiment

from bnn import BNN
from adaptive_sghmc import AdaptiveSGHMC

tfs = tf.summary
tfd = tfp.distributions

tfl = tf.keras.layers

ex = Experiment("bvae_experiment", ingredients=[])


@ex.config
def config():
    data_dir = "/scratch/gf332/Misc/datasets/"
    model_base_save_dir = "/scratch/gf332/Misc/bnn_experiments"

    batch_size = 500
    burnin_epochs = 50

    dataset_size = 60000

    # Gradient descent Optimizer
    iterations = 100000
    learning_rate = 1e-2

    # Logging
    tensorboard_log_freq = 100

    model_save_dir = f"{model_base_save_dir}/bnn"

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"{model_save_dir}/logs/{current_time}/train"


@ex.automain
def train(model_save_dir,
          data_dir,
          dataset_size,

          batch_size,
          burnin_epochs,

          iterations,
          learning_rate,

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
    train_data = train_data.map(lambda x: (tf.cast(x["image"], tf.float32) / 255., x["label"]))
    train_data = train_data.shuffle(5000)
    train_data = train_data.repeat()
    train_data = train_data.batch(batch_size)
    train_data = train_data.prefetch(10)

    # -------------------------------------------------------------------------
    # Create model
    # -------------------------------------------------------------------------

    model = BNN()

    model.build(input_shape=(batch_size, 28, 28, 1))

    learning_rate = tf.Variable(learning_rate, dtype=tf.float32, name="learn_rate")

    optimizer = AdaptiveSGHMC(learning_rate=learning_rate,
                              burnin=num_batch_per_epoch * burnin_epochs,
                              data_size=dataset_size,
                              momentum_decay=0.05)

    ckpt = tf.train.Checkpoint(step=tf.Variable(1, dtype=tf.int64),
                               learning_rate=learning_rate,
                               model=model,
                               optimizer=optimizer)

    manager = tf.train.CheckpointManager(ckpt, model_save_dir, max_to_keep=5)

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

    def train_step(model, batch, labels):

        with tf.GradientTape() as tape:
            probabilities = model(batch)

            log_likelihood = tf.reduce_mean(labels * tf.math.log(probabilities))

            prior_log_prob = model.weight_prior_log_prob() / tf.cast(dataset_size, tf.float32)
            hyper_prior_log_prob = model._hyperprior_log_prob / tf.cast(dataset_size, tf.float32)

            joint_log_likelihood = log_likelihood + prior_log_prob + hyper_prior_log_prob

        gradients = tape.gradient(joint_log_likelihood, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if any(tf.reduce_any(tf.math.is_nan(grad)) for grad in gradients):
            raise ValueError(f"Gradient exploded: LL: {log_likelihood}, Prior LL: {prior_log_prob}, Hyperprior LL: {hyper_prior_log_prob}")

        return probabilities, log_likelihood, prior_log_prob, hyper_prior_log_prob

    for batch, labels in train_data:

        # Resample scale hyperparameters every epoch
        if int(ckpt.step) % num_batch_per_epoch == 0:
            _log.info("Resampling hyperparameters!")

            model.resample_weight_prior_parameters()

        ckpt.step.assign_add(1)

        probabilities, log_likelihood, prior_log_prob, hyper_prior_log_prob = train_step(model, batch, tf.one_hot(labels, depth=10))

        prediction = tf.argmax(probabilities, axis=1)

        accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, labels), tf.float32))

        if int(ckpt.step) % num_batch_per_epoch == 0:
            # Save model
            save_path = manager.save()
            _log.info(f"Step {int(ckpt.step)}: Saved model to {save_path}")

            with summary_writer.as_default():
                tfs.scalar(name="Joint_Log_Likelihood", data=log_likelihood + prior_log_prob + hyper_prior_log_prob,
                           step=ckpt.step)
                tfs.scalar(name="Data_log_likelihood", data=log_likelihood, step=ckpt.step)
                tfs.scalar(name="Prior_log_prob", data=prior_log_prob, step=ckpt.step)
                tfs.scalar(name="Hyperprior_log_prob", data=hyper_prior_log_prob, step=ckpt.step)
                tfs.scalar(name="Train_accuracy", data=accuracy, step=ckpt.step)
