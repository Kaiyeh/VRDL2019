import helper as hp
import tensorflow as tf
import os
import numpy as np
from PIL import Image

batch_size = 9
noise_size = 100
sl5 = 4
img_size = 64
rinit = tf.random_normal_initializer(stddev=0.02)

# Generator def
G = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units = 512*sl5*sl5, input_dim=noise_size, kernel_initializer=rinit),
    tf.keras.layers.Reshape(target_shape=[sl5,sl5,512]),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(),

    tf.keras.layers.UpSampling2D(),
    tf.keras.layers.Conv2D(filters=256, kernel_size=5, strides=1, padding='same', kernel_initializer=rinit),
    tf.keras.layers.BatchNormalization(momentum=0.8),
    tf.keras.layers.LeakyReLU(),

    tf.keras.layers.UpSampling2D(),
    tf.keras.layers.Conv2D(filters=128, kernel_size=5, strides=1, padding='same', kernel_initializer=rinit),
    tf.keras.layers.BatchNormalization(momentum=0.8),
    tf.keras.layers.LeakyReLU(),

    tf.keras.layers.UpSampling2D(),
    tf.keras.layers.Conv2D(filters=64, kernel_size=5, strides=1, padding='same', kernel_initializer=rinit),
    tf.keras.layers.BatchNormalization(momentum=0.8),
    tf.keras.layers.LeakyReLU(),

    tf.keras.layers.UpSampling2D(),
    tf.keras.layers.Conv2D(filters=3, kernel_size=5, strides=1, padding='same', activation='tanh', kernel_initializer=rinit)
    ])

if __name__ == '__main__':
    G.summary()

    lo_it = 3
    lo_ba = 0
    pfix = '-iter-' + str(lo_it) + '-' + str(lo_ba) + '.0'
    try:
        G.load_weights('./weights/G/G' + pfix)
    except:
        print('G weights not found!')
        input()

    G.summary()
    for i in range(500):
        nd_noise = np.random.normal(0, 1, (batch_size, noise_size))
        #genimg = ( * 127.5) + 127.5
        gridimg = hp.images_square_grid(G.predict(nd_noise))
        gridimg.save('imggen/' + ('%03d' % i) + '_image.png')
    '''
    img_size = 64
    batch_size = 9
    path = '../dataset/img_align_celeba/'
    dir = np.array([path + fname for fname in os.listdir(path)])
    select = np.random.randint(0, dir.shape[0], batch_size)
    real_image = (hp.get_batch(dir[select], img_size, img_size, 'RGB') - 127.5) / 127.5
    imgg = hp.images_square_grid(real_image)
    imgg.save('grid.png')
    '''