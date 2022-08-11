import numpy as np
from tensorflow import keras
from mnist_vae import VAE, create_encoder, create_decoder

(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
mnist_digits = np.concatenate([x_train, x_test], axis=0)
mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255

latent_dim = 2
encoder = create_encoder(latent_dim)
decoder = create_decoder(latent_dim)
vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam(), run_eagerly=True)

history = vae.fit(
    mnist_digits,
    epochs=30,
    batch_size=128,
    callbacks=[
        keras.callbacks.ModelCheckpoint(
            "mnist_vae.v3.keras",
            monitor="total_loss",
            save_best_only=True,
            save_weights_only=True,
        )
    ],
)
