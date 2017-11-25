import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import argparse
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data

# Choose dataset
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default='mnist', help="Options: mnist or fashio-mnist")
args = parser.parse_args()

# Noise dimensions
zdim = 50

# Samples
Nsamples = 100

# Input to the discriminator
x = tf.placeholder(tf.float32, shape=[None, 784])

# Input to the generator
z = tf.placeholder(tf.float32, shape=[None, zdim])

# Load data
if args.dataset is 'mnist':
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
else:
    mnist = input_data.read_data_sets('Fashion-MNIST_data', one_hot=True,
                                  source_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/')

# Define the generator
def generator(input, is_training=True, reuse=False):
    with tf.variable_scope("generator", reuse=reuse):
        net = slim.fully_connected(input, 128, activation_fn=tf.nn.relu, trainable=is_training, reuse=reuse, scope='gen1')
        net = slim.fully_connected(net, 784, activation_fn=tf.sigmoid, trainable=is_training, reuse=reuse, scope='gen2')
        return net

# Define the discriminator
def discriminator(input, is_training=True, reuse=False):
    with tf.variable_scope("discriminator", reuse=reuse):
        net = slim.fully_connected(input, 128, activation_fn=tf.nn.relu, trainable=is_training, reuse=reuse, scope='dic1')
        logit = slim.fully_connected(net, 1, activation_fn=None, trainable=is_training, reuse=reuse, scope='dic2')
        prob = tf.nn.sigmoid(logit)
        return logit, prob

# Generate samples
def draw_sample(m, n):
    return np.random.uniform(-1.0, 1.0, size=[m, n])

# Inference: Generate Sample, Classify real / generated sample
G_sample_train = generator(z)
D_logit_real, D_real = discriminator(x)
D_logit_gen, D_fake = discriminator(G_sample_train, reuse=True)
G_sample_inf = generator(z, is_training=False, reuse=True)

# Loss Functions
D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real))) + \
         tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_gen, labels=tf.zeros_like(D_logit_gen)))
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_gen, labels=tf.ones_like(D_logit_gen)))

# All variables
t_vars = tf.trainable_variables()

# Generator variables (used during G update)
g_vars = [var for var in t_vars if 'generator' in var.name]

# Discriminator variables (used during D update)
d_vars = [var for var in t_vars if 'discriminator' in var.name]

# Optimizers
D_opt = tf.train.AdamOptimizer().minimize(loss=D_loss, var_list=d_vars)
G_opt = tf.train.AdamOptimizer().minimize(loss=G_loss, var_list=g_vars)

with tf.Session() as sess:

    # Initialize all variables
    tf.global_variables_initializer().run()

    # training iterations
    for i in range(int(1e6)):
        imgs, labels = mnist.train.next_batch(Nsamples)

        # 1. Update the discriminator using real and generated samples
        feed_dict_d = {x: imgs, z: draw_sample(Nsamples, zdim)}
        _, D_loss_val = sess.run([D_opt, D_loss], feed_dict=feed_dict_d)

        # 2. Update the generator
        feed_dict_g = {z: draw_sample(Nsamples, zdim)}
        _, G_loss_val = sess.run([G_opt, G_loss], feed_dict=feed_dict_g)

        if i % int(1e3) == 0:
            print('Iteration {}, Discriminator Loss {:.3}, Generator Loss {:.3}'.format(i, D_loss_val, G_loss_val))

            # sample from the generator
            sample = sess.run(G_sample_inf, feed_dict=feed_dict_g)

            # Visualise the sampled images
            fig = plt.figure(figsize=(10, 10))
            gs = gridspec.GridSpec(10, 10)
            for j, sample in enumerate(sample):
                ax = plt.subplot(gs[j])
                ax.set_xticks([])
                ax.set_yticks([])
                plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
            if not os.path.exists('res'):
                os.makedirs('res')
            plt.savefig('res/{}.png'.format(str(i).zfill(7)), bbox_inches='tight')
            plt.close()