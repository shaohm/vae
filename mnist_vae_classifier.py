import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def create_encoder(latent_dim):
    encoder_inputs = keras.Input(shape=(28, 28, 1))
    x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(
        encoder_inputs
    )  # 14, 14, 32
    x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(
        x
    )  # 7, 7, 64
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_logvar = layers.Dense(latent_dim, name="z_logvar")(x)
    encoder = keras.Model(encoder_inputs, [z_mean, z_logvar], name="encoder")
    return encoder


class Sampler(layers.Layer):
    def call(self, z_mean, z_logvar):
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_logvar) * epsilon


def create_decoder(latent_dim):
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
    x = layers.Reshape((7, 7, 64))(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(
        x
    )  # 14, 14, 64
    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(
        x
    )  # 28, 28, 32
    decoder_outputs = layers.Conv2D(
        1, 3, strides=1, padding="same", activation="sigmoid"
    )(x)

    x = layers.Dense(64, activation="relu")(latent_inputs)
    predict_outputs = layers.Dense(10, activation="softmax")(x)

    decoder = keras.Model(
        inputs=latent_inputs, outputs=[decoder_outputs, predict_outputs], name="decoder"
    )
    return decoder


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.sampler = Sampler()
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.classification_loss_tracker = keras.metrics.Mean(
            name="classification_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.classification_loss_tracker,
            self.kl_loss_tracker,
        ]

    def call(self, data):
        z_mean, z_logvar = self.encoder(data)
        z = self.sampler(z_mean, z_logvar)
        r, y = self.decoder(z)
        return r, y

    def train_step(self, data):
        data, y_true = data
        with tf.GradientTape() as tape:
            z_mean, z_logvar = self.encoder(data)
            z = self.sampler(z_mean, z_logvar)
            r, y = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(keras.losses.binary_crossentropy(data, r), axis=(1, 2))
            )

            classification_loss = keras.losses.SparseCategoricalCrossentropy(
                reduction=tf.keras.losses.Reduction.SUM
            )(y_true, y)
            # kl_loss = tf.reduce_mean(- 0.5 * tf.reduce_sum(1 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar))), #未加axis
            kl_loss = tf.reduce_mean(
                tf.reduce_sum(
                    -0.5 * (1 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar)), axis=1
                )
            )
            # kl_loss = - 0.5 * (1 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar))
            total_loss = reconstruction_loss + classification_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.classification_loss_tracker.update_state(classification_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "classification_loss": self.classification_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
