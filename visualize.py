import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from mnist_vae import VAE, create_encoder, create_decoder

latent_dim = 2
encoder = create_encoder(latent_dim)
decoder = create_decoder(latent_dim)
vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam(), run_eagerly=True)

sample_x = tf.random.uniform((1, 28, 28, 1))
vae(sample_x)

#vae.load_weights("mnist_vae_classifier.v0.keras")
vae.load_weights(sys.argv[1])

import matplotlib.pyplot as plt

n = 30
digit_size = 28

figure = np.zeros((digit_size * n, digit_size * n))

grid_x = np.linspace(-1, 1, n)
grid_y = np.linspace(-1, 1, n)[::-1]

for i, yi in enumerate(grid_y):
    for j, xi in enumerate(grid_x):
        z_sample = np.array([[xi, yi]])
        x_decoded = vae.decoder.predict(z_sample, verbose=0)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[
            i * digit_size : (i + 1) * digit_size, j * digit_size : (j + 1) * digit_size
        ] = digit
start_range = digit_size // 2
end_range = n * digit_size + start_range
pixel_range = np.arange(start_range, end_range, digit_size)
sample_range_x = np.round(grid_x, 1)
sample_range_y = np.round(grid_y, 1)
plt.xticks(pixel_range, sample_range_x)
plt.yticks(pixel_range, sample_range_y)
plt.xlabel("z[0]")
plt.ylabel("z[1]")
plt.axis("off")
plt.imshow(figure, cmap="Greys_r")
plt.show()
