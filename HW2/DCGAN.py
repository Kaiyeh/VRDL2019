import helper as hp
import tensorflow as tf
import os
import numpy as np
import random
from PIL import Image
import tensorflow.keras.backend as K
from functools import partial

path = '../dataset/img_align_celeba/'
hp.download_extract('../dataset')

batch_size = 64
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
# Discriminator def
D = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=64,kernel_size=5,strides=2,padding='same', input_shape=(img_size,img_size,3), kernel_initializer=rinit),
    tf.keras.layers.LeakyReLU(alpha=0.2),

    tf.keras.layers.Conv2D(filters=128,kernel_size=5,strides=2,padding='same', kernel_initializer=rinit),
    tf.keras.layers.BatchNormalization(momentum=0.8),
    tf.keras.layers.LeakyReLU(alpha=0.2),

    tf.keras.layers.Conv2D(filters=256,kernel_size=5,strides=2,padding='same', kernel_initializer=rinit),
    tf.keras.layers.BatchNormalization(momentum=0.8),
    tf.keras.layers.LeakyReLU(alpha=0.2),

    tf.keras.layers.Conv2D(filters=512, kernel_size=5, strides=2, padding='same', kernel_initializer=rinit),
    tf.keras.layers.BatchNormalization(momentum=0.8),
    tf.keras.layers.LeakyReLU(alpha=0.2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(units = 1, activation='sigmoid', kernel_initializer=rinit)])


def dcmat(y_true, y_pred):
    return K.sum(y_pred)

if __name__ == '__main__':
    D.summary()
    G.summary()

    lo_it = 6
    lo_ba = 0
    pfix = '-iter-' + str(lo_it) + '-' + str(lo_ba) + '.0'
    try:
        D.load_weights('./weights/D/D' + pfix)
    except:
        print('D weights not found!')
        input()
    D.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5), metrics=[dcmat])
    try:
        G.load_weights('./weights/G/G' + pfix)
    except:
        print('G weights not found!')
        input()

    G.summary()
    D.summary()

    z = tf.keras.layers.Input(shape=(noise_size))


    D.trainable = False
    D_fake = D(G(z))
    G_train = tf.keras.models.Model(inputs=[z], outputs=[D_fake], name="Combine")
    G_train.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5), metrics=[dcmat])
    G_train.summary()

    dir = np.array([path+fname for fname in os.listdir(path)])
    print('Total number of images: ',len(dir))
    maxepch = 25
    tones = np.ones((batch_size, 1))
    ones = np.ones((batch_size,1)) - 0.01
    zeros = np.zeros((batch_size,1))

    dummies = np.ones((batch_size))

    accnt = 0
    dacr_sum = np.zeros(2)
    dacf_sum = np.zeros(2)
    gac_sum = np.zeros(2)

    for epch in range(maxepch):
        print('Iteration No.', epch)
        for bcnt in range(0, dir.shape[0], batch_size):
            #random.shuffle(dir)
            n_critic = 1
            for ct in range(n_critic):
                # Label smoothing
                #label_r = ones + (np.random.sample([batch_size,1]) * 0.01)
                #label_f = zeros + (np.random.sample([batch_size, 1]) * 0.01)

                select = np.random.randint(0, dir.shape[0], batch_size)
                real_image = (hp.get_batch(dir[select], img_size, img_size, 'RGB') - 127.5) / 127.5
                # nd_noise = np.random.normal(0, 1, (batch_size, noise_size))
                nd_noise = np.random.uniform(-1, 1, (batch_size, noise_size))
                fake_img = G.predict(nd_noise)

                dacr_sum += D.train_on_batch(real_image, tones)
                dacf_sum += D.train_on_batch(fake_img, zeros)

            # Train generator
            # nd_noise = np.random.normal(0,1,(batch_size, noise_size))
            nd_noise = np.random.uniform(-1, 1, (batch_size, noise_size))
            gac_sum += G_train.train_on_batch(nd_noise, tones)
            accnt += 1
            if(bcnt%4992==0):
                print('Batch no.', bcnt)
                print('Discriminator ave real loss:', dacr_sum / accnt / n_critic)
                print('Discriminator ave fake loss:', dacf_sum / accnt / n_critic)
                print('Discriminator ave loss:', (dacr_sum[0] + dacf_sum[0]) / accnt / n_critic)
                print('Generator ave loss: %f %f' % tuple(gac_sum / accnt))

                # nd_noise = np.random.normal(0, 1, (batch_size, noise_size))
                nd_noise = np.random.uniform(-1, 1, (batch_size, noise_size))
                fake_img = G.predict(nd_noise)
                for i in range(int(bcnt/9984),20):
                    img = Image.fromarray((fake_img[i] * 127.5) + 127.5, 'RGB')
                    img.save('outputs/' + 'iter-' + str(epch) + '-' + str(i) + '.png')
                if(bcnt%(9984*10)==0):
                    D.save_weights('./weights/D/D' + '-iter-' + str(epch) + '-' + str(bcnt/(9984*10)))
                    G.save_weights('./weights/G/G' + '-iter-' + str(epch) + '-' + str(bcnt/(9984*10)))
                    with open('log.txt','a') as w:
                        w.write('Iter-' + str(epch) + '-' + str(bcnt/(9984*10)) + '\n')
                        w.write(str(dacr_sum / accnt / n_critic) + '\n')
                        w.write(str(dacf_sum / accnt / n_critic) + '\n')
                        w.write(str((dacr_sum[0] + dacf_sum[0]) / accnt / n_critic) + '\n')
                        w.write(str(gac_sum / accnt) + '\n')
                    print('Model saved!')

                accnt = 0
                dacr_sum = np.zeros(2)
                dacf_sum = np.zeros(2)
                gac_sum = np.zeros(2)


