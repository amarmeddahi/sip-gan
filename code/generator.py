# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 14:17:19 2021

@author: Amar Meddahi (amar.meddahi1[at]gmail.com)
"""

import os
import time
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from toolbox import *

# Load and prepare the dataset
nb_packets = 20000
multimap = 2
image_size = optimal_image_size(multimap)

dir_images = 'images'
images_path = os.path.abspath(dir_images) + '\\'

dir_generated = 'generated'
os.mkdir(dir_generated)
generated_path = os.path.abspath(dir_generated) + '\\'

train_images = images_to_tensor_slices(images_path, nb_packets)
train_images = train_images.reshape(train_images.shape[0], image_size[0], image_size[1], 1).astype('float32')
train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

BUFFER_SIZE = nb_packets
BATCH_SIZE = 32

# Batch and shuffle the data
train_images = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# The Generator
def make_generator_model(multimap):
    model = tf.keras.Sequential()
    model.add(layers.Dense(10*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((10, 7, 256)))
    assert model.output_shape == (None, 10, 7, 256)  # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 10, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 20, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 40, 28, 32)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    if multimap >= 4:
        model.add(layers.Conv2DTranspose(16, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 80, 56, 16)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        if multimap >= 8:
            model.add(layers.Conv2DTranspose(8, (5, 5), strides=(2, 2), padding='same', use_bias=False))
            assert model.output_shape == (None, 160, 112, 8)
            model.add(layers.BatchNormalization())
            model.add(layers.LeakyReLU())
            if multimap >= 16:
                model.add(layers.Conv2DTranspose(8, (5, 5), strides=(2, 2), padding='same', use_bias=False))
                assert model.output_shape == (None, 320, 224, 8)
                model.add(layers.BatchNormalization())
                model.add(layers.LeakyReLU())
                if multimap == 32:
                    model.add(layers.Conv2DTranspose(8, (5, 5), strides=(2, 2), padding='same', use_bias=False))
                    assert model.output_shape == (None, 640, 448, 8)
                    model.add(layers.BatchNormalization())
                    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, image_size[0], image_size[1], 1)

    return model

generator = make_generator_model(multimap)

# The Discriminator
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[image_size[0], image_size[1], 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

discriminator = make_discriminator_model()

# Test
def gan_test(model, test_input, param):
    dirTestName = generated_path + str(param)
    os.mkdir(dirTestName)

    predictions = model(test_input, training=False)
    predictions = predictions.numpy()
    predictions = predictions.reshape(predictions.shape[0], predictions.shape[1], predictions.shape[2])
    path = os.path.abspath(dirTestName) + '\\'

    for i in range(predictions.shape[0]):
        I = predictions[i, :, :] * 127.5 + 127.5
        im = Image.fromarray(I.astype(np.uint8))
        im.save(path + str(i) + ".png", 'PNG')

# Define the loss and optimizers
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Training loop
eps = 0.0001 # epsilon
noise_dim = 100

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(dataset, eps):
    total_time = 0
    n_generated = 1000 # number of generated images to compute the quality factor gamma
    n_test = 1000 # number of test done to compute the quality factor
    max_iter = 1000   # max iterations (K)
    epoch = 0
    numerical_data = np.zeros((2,max_iter)) # (epoch, gamma)
    while True:
        
        # Step 1: a single epoch training
        start = time.time()
        for image_batch in dataset:
            train_step(image_batch)

        # Step 2: compute the quality factor gamma
        seed = tf.random.normal([n_generated, noise_dim])
        gan_test(generator, seed, epoch + 1)
        gamma = stats_error(images_path, nb_packets, generated_path + str(epoch+1), n_generated, multimap, n_test)
        numerical_data[:,epoch] =  [epoch,gamma]
        print ('Gamma for epoch {} is {}'.format(epoch + 1, gamma))
        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
        total_time += time.time()-start
        epoch += 1
    
        # Step 3: stop condition
        if gamma <= eps or epoch >= max_iter:
            break
    print('Time for ' + str(epoch) + ' epochs is ' + str(total_time) + ' sec')
    return numerical_data

compute_data = train(train_images, eps)
