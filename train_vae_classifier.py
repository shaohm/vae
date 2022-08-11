import numpy as np
from tensorflow import keras
from mnist_vae_classifier import VAE, create_encoder, create_decoder

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = np.expand_dims(x_train, -1).astype("float32") / 255
x_test = np.expand_dims(x_test, -1).astype("float32") / 255
y_train = np.array(y_train).astype("float32")
y_test = np.array(y_test).astype("float32")
print(y_train[:10])

latent_dim = 2
encoder = create_encoder(latent_dim)
decoder = create_decoder(latent_dim)
vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())
# vae.compile(optimizer=keras.optimizers.Adam(), run_eagerly=True)

history = vae.fit(
    x_train,
    y_train,
    epochs=30,
    batch_size=128,
    validation_data=(x_test, y_test),
    callbacks=[
        keras.callbacks.ModelCheckpoint(
            "mnist_vae_classifier.v1.keras",
            monitor="total_loss",
            save_best_only=True,
            save_weights_only=True,
        )
    ],
)
