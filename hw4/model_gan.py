import os, re, time, glob
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import batch_norm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import skimage
import skimage.transform
import skimage.io

import pandas as pd
from sklearn.metrics import accuracy_score

from ops import *
from utils import *

class GAN(object):
    def __init__(self,
                 sess,
                 model_name='GAN',
                 result_path='/data/put_data/cclin/ntu/dlcv2018/hw4/results',
                 random_dim=100,
                 img_size=64,
                 c_dim=3,
                 bnDecay=0.9,
                 epsilon=1e-5):
        self.sess = sess
        self.model_name = model_name
        self.result_path = result_path
        self.random_dim = random_dim
        self.img_size = img_size
        self.c_dim = c_dim
        self.bnDecay = bnDecay
        self.epsilon = epsilon
    
    # Build the GAN
    def build_model(self):
        image_dims = [self.img_size, self.img_size, self.c_dim]
        self.input_images = tf.placeholder(tf.float32, shape=[None]+image_dims, name='input_images')
        self.z_random = tf.placeholder(tf.float32, shape=[None, self.random_dim], name='latent_vec')
        self.bn_train = tf.placeholder('bool')
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        
        ## batch normalization
        self.d_bn0 = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='d_bn0')
        self.d_bn1 = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='d_bn1')
        self.d_bn2 = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='d_bn2')
        self.d_bn3 = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='d_bn3')
        self.g_bn0 = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='g_bn0')
        self.g_bn1 = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='g_bn1')
        self.g_bn2 = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='g_bn2')
        self.g_bn3 = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='g_bn3')
        
        ## training data operations
        self.image_fake = self.generator(self.z_random, self.bn_train)
        self.d_logit_real = self.discriminator(self.input_images, self.bn_train)
        self.d_logit_fake = self.discriminator(self.image_fake, self.bn_train, reuse=True)
        self.image_sample = self.generator(self.z_random, bn_train=False, reuse=True)
        
        ## loss
        self.loss_d_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_logit_real,
                                                                                  labels=tf.ones_like(self.d_logit_real)))
        self.loss_d_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_logit_fake,
                                                                                  labels=tf.zeros_like(self.d_logit_fake)))
        self.loss_d = self.loss_d_real + self.loss_d_fake
        self.loss_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_logit_fake,
                                                                             labels=tf.ones_like(self.d_logit_fake)))
        ## training operations
        train_vars = tf.trainable_variables()
        self.vars_d = [var for var in train_vars if 'discriminator' in var.name]
        self.vars_g = [var for var in train_vars if 'generator' in var.name]
        self.train_op_d = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                 beta1=0.5).minimize(self.loss_d,
                                                                     var_list=self.vars_d)
        self.train_op_g = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                 beta1=0.5).minimize(self.loss_g,
                                                                     var_list=self.vars_g)
        
        ## Create model saver (keep all checkpoint!)
        self.saver = tf.train.Saver(max_to_keep = None)
       
    def discriminator(self, input_images, bn_train, reuse = False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
            h0 = lrelu(self.d_bn0(conv2d(input_images, output_dim=32, name='h0'), train=bn_train), leak=0.1) ## [-1, 32, 32, 32]
            h1 = lrelu(self.d_bn1(conv2d(h0, output_dim=64, name='h1'), train=bn_train), leak=0.1) ## [-1, 16, 16, 64]
            h2 = lrelu(self.d_bn2(conv2d(h1, output_dim=128, name='h2'), train=bn_train), leak=0.1) ## [-1, 8, 8, 128]
            h3 = lrelu(self.d_bn3(conv2d(h2, output_dim=256, name='h3'), train=bn_train), leak=0.1) ## [-1, 4, 4, 256]
            h4 = linear(tf.reshape(h3, [-1, 4096]), 1, 'h4')
            return h4
    
    def generator(self, z_random, bn_train, reuse = False):
        with tf.variable_scope("generator") as scope:
            if reuse:
                scope.reuse_variables()
            bsize = tf.shape(z_random)[0]
            h0 = tf.reshape(linear(z_random, 4096, 'h0'), [-1, 4, 4, 256])
            h0 = tf.nn.relu(self.g_bn0(h0, train=bn_train))
            h1 = deconv2d(h0, [bsize, 8, 8, 128], name='h1')
            h1 = tf.nn.relu(self.g_bn1(h1, train=bn_train))
            h2 = deconv2d(h1, [bsize, 16, 16, 64], name='h2')
            h2 = tf.nn.relu(self.g_bn2(h2, train=bn_train))
            h3 = deconv2d(h2, [bsize, 32, 32, 32], name='h3')
            h3 = tf.nn.relu(self.g_bn3(h3, train=bn_train))
            h4 = deconv2d(h3, [bsize, 64, 64, 3], name='h4')
            return (tf.tanh(h4)/2. + 0.5)
    
    def train(self,
              init_from=None,
              train_path='/data/put_data/cclin/ntu/dlcv2018/hw4/hw4_data/train',
              test_path='/data/put_data/cclin/ntu/dlcv2018/hw4/hw4_data/test',
              train_period_d=1,
              train_period_g=2,
              nEpochs=200,
              bsize=32,
              learning_rate_start=1e-3,
              patience=10):
        ## create a dedicated folder for this model
        if os.path.exists(os.path.join(self.result_path, self.model_name)):
            print('WARNING: the folder "{}" already exists!'.format(os.path.join(self.result_path, self.model_name)))
        else:
            os.makedirs(os.path.join(self.result_path, self.model_name))
            os.makedirs(os.path.join(self.result_path, self.model_name, 'samples'))
            os.makedirs(os.path.join(self.result_path, self.model_name, 'models'))
        
        ## data list (it is allowed to use both training and testing images)
        data_list = np.sort(glob.glob(os.path.join(train_path, '*.png')))
        if test_path is not None:
            data_list = np.concatenate((data_list, np.sort(glob.glob(os.path.join(test_path, '*.png')))), axis=0)
        nBatches = int(np.ceil(len(data_list) / bsize))
        
        ## initialization
        initOp = tf.global_variables_initializer()
        self.sess.run(initOp)
        
        ## main training loop
        loss_d_list = []
        loss_g_list = []
        best_vae_loss = 0
        stopping_step = 0
        for epoch in range(1, (nEpochs+1)):
            loss_d_batch = []
            loss_g_batch = []
            idx = 0
            np.random.shuffle(data_list)
            while idx < nBatches:
                #### update D 
                for _ in range(train_period_d):
                    batch_files = data_list[idx*bsize:(idx+1)*bsize]
                    if len(batch_files) == 0:
                        idx = nBatches
                        break
                    batch = [get_image(batch_file) for batch_file in batch_files]
                    batch_images = np.array(batch).astype(np.float32)
                    batch_z_random = np.random.uniform(-1, 1, [batch_images.shape[0], self.random_dim]).astype(np.float32)
                    _, loss_d = self.sess.run([self.train_op_d, self.loss_d],
                                              feed_dict={self.input_images: batch_images,
                                                         self.z_random: batch_z_random,
                                                         self.bn_train: True,
                                                         self.learning_rate: learning_rate_start})
                    loss_d_batch.append(loss_d)
                    idx += 1
                #### update G
                for _ in range(train_period_g):
                    batch_z_random = np.random.uniform(-1, 1, [bsize, self.random_dim]).astype(np.float32)
                    _, loss_g = self.sess.run([self.train_op_g, self.loss_g],
                                              feed_dict={self.z_random: batch_z_random,
                                                         self.bn_train: True,
                                                         self.learning_rate: learning_rate_start})
                    loss_g_batch.append(loss_g)
            ### record D and G loss for each iteration (instead of each epoch)
            loss_d_list.extend(loss_d_batch)
            loss_g_list.extend(loss_g_batch)
            # loss_d.append(np.mean(loss_d_batch))
            # loss_g.append(np.mean(loss_g_batch))
            
            print('Epoch: %d, loss_d: %f, loss_g: %f' % \
                  (epoch, np.mean(loss_d_batch), np.mean(loss_g_batch)))
            
            ### save model and run inference for every 10 epochs
            if epoch % 10 == 0:
                #### produce 32 random images
                batch_z_random = np.random.uniform(-1, 1, [32, self.random_dim]).astype(np.float32)
                samples = self.sess.run(self.image_sample, feed_dict={self.z_random: batch_z_random,
                                                                      self.bn_train: False})
                fig = self.plot(samples, 4, 8)
                plt.savefig(os.path.join(self.result_path, self.model_name, 'samples', '{}.png'.format(str(epoch).zfill(3))), 
                            bbox_inches='tight')
                plt.close(fig)
                #### save model only if epoch >= 100 (more stable)
                if epoch >= 100:
                    self.saver.save(self.sess,
                                    os.path.join(self.result_path, self.model_name, 'models', self.model_name + '.model'),
                                    global_step=epoch)
            
        return [loss_d_list, loss_g_list]
    
    def inference(self,
                  gen_from=None,
                  gen_from_ckpt=None,
                  out_path=None,
                  bsize=32,
                  set_seed=1002):
        ## create output folder
        if gen_from is None:
            gen_from = os.path.join(self.result_path, self.model_name, 'models')
        if out_path is not None:
            if os.path.exists(out_path):
                print('WARNING: the output path "{}" already exists!'.format(out_path))
            else:
                os.makedirs(out_path)
        else:
            out_path = os.path.join(self.result_path, self.model_name)
        
        ## load previous model if possible
        could_load, checkpoint_counter = self.load(gen_from, gen_from_ckpt)
        if could_load:
            print(" [*] Load SUCCESS")
            ### set seed to generate identical figures for the report
            np.random.seed(set_seed)
            
            ### fig2_2.jpg: learning curve
            #### Assume that 'results.npy' (containing losses vs. training iterations)
            #### is saved in the same directory as the model checkpoint
            results = np.load(os.path.join(gen_from, 'results.npy'))
            fig, ax = plt.subplots(1,2, figsize=(16,6))
            ax[0].plot(range(len(results[0])), results[0])
            ax[0].set_xlabel('Training iterations', fontsize=16)
            ax[0].set_title('D loss', fontsize=20)
            ax[1].plot(range(len(results[1])), results[1])
            ax[1].set_xlabel('Training iterations', fontsize=16)
            ax[1].set_title('G loss', fontsize=20)
            plt.savefig(os.path.join(out_path, 'fig2_2.jpg'))
            plt.close(fig)
            
            #### fig2_3.jpg: produce 32 random images
            batch_z_random = np.random.uniform(-1, 1, [32, self.random_dim]).astype(np.float32)
            samples = self.sess.run(self.image_sample, feed_dict={self.z_random: batch_z_random,
                                                                  self.bn_train: False})
            fig = self.plot(samples, 4, 8)
            plt.savefig(os.path.join(out_path, 'fig2_3.jpg'), bbox_inches='tight')
            plt.close(fig)
    
    def plot(self, samples, n_row, n_col):
        fig = plt.figure(figsize=(n_col*2, n_row*2))
        gs = gridspec.GridSpec(n_row, n_col)
        gs.update(wspace=0.05, hspace=0.05)
        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(64, 64, 3))
        return fig
    
    def load(self, init_from, init_from_ckpt=None):
        ckpt = tf.train.get_checkpoint_state(init_from)
        if ckpt and ckpt.model_checkpoint_path:
            if init_from_ckpt is None:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            else:
                ckpt_name = init_from_ckpt
            self.saver.restore(self.sess, os.path.join(init_from, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

class WGAN(GAN):
    def __init__(self,
                 sess,
                 model_name='WGAN',
                 result_path='/data/put_data/cclin/ntu/dlcv2018/hw4/results',
                 random_dim=100,
                 img_size=64,
                 c_dim=3,
                 bnDecay=0.9,
                 epsilon=1e-5,
                 gp_scale=10.0):
        super(WGAN, self).__init__(sess,
                                   model_name,
                                   result_path,
                                   random_dim,
                                   img_size,
                                   c_dim,
                                   bnDecay,
                                   epsilon)
        self.gp_scale = gp_scale
    
    # Only need to re-define build_model()
    def build_model(self):
        image_dims = [self.img_size, self.img_size, self.c_dim]
        self.input_images = tf.placeholder(tf.float32, shape=[None]+image_dims, name='input_images')
        self.z_random = tf.placeholder(tf.float32, shape=[None, self.random_dim], name='latent_vec')
        self.bn_train = tf.placeholder('bool')
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        
        ## batch normalization
        self.d_bn0 = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='d_bn0')
        self.d_bn1 = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='d_bn1')
        self.d_bn2 = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='d_bn2')
        self.d_bn3 = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='d_bn3')
        self.g_bn0 = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='g_bn0')
        self.g_bn1 = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='g_bn1')
        self.g_bn2 = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='g_bn2')
        self.g_bn3 = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='g_bn3')
        
        ## training data operations
        self.image_fake = self.generator(self.z_random, self.bn_train)
        self.d_logit_real = self.discriminator(self.input_images, self.bn_train)
        self.d_logit_fake = self.discriminator(self.image_fake, self.bn_train, reuse=True)
        self.image_sample = self.generator(self.z_random, bn_train=False, reuse=True)
        
        ## loss
        self.loss_d_real = tf.reduce_mean(self.d_logit_real)
        self.loss_d_fake = tf.reduce_mean(self.d_logit_fake)
        self.loss_d = self.loss_d_fake - self.loss_d_real
        self.loss_g = -tf.reduce_mean(self.d_logit_fake)
        
        ## gradient penalty
        epsilon = tf.random_uniform([], 0.0, 1.0)
        x_hat = epsilon * self.input_images + (1 - epsilon) * self.image_fake
        d_hat = self.discriminator(x_hat, self.bn_train, reuse=True)
        self.ddx = tf.gradients(d_hat, x_hat)[0]
        self.ddx = tf.sqrt(tf.reduce_sum(tf.square(self.ddx), axis=1))
        self.ddx = tf.reduce_mean(tf.square(self.ddx - 1.0) * self.gp_scale)
        self.loss_d = self.loss_d + self.ddx
        
        ## training operations
        train_vars = tf.trainable_variables()
        self.vars_d = [var for var in train_vars if 'discriminator' in var.name]
        self.vars_g = [var for var in train_vars if 'generator' in var.name]
        self.train_op_d = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                 beta1=0.5).minimize(self.loss_d,
                                                                     var_list=self.vars_d)
        self.train_op_g = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                 beta1=0.5).minimize(self.loss_g,
                                                                     var_list=self.vars_g)
        
        ## Create model saver (keep all checkpoint!)
        self.saver = tf.train.Saver(max_to_keep = None)

class ACGAN(GAN):
    def __init__(self,
                 sess,
                 model_name='ACGAN',
                 result_path='/data/put_data/cclin/ntu/dlcv2018/hw4/results',
                 random_dim=100,
                 img_size=64,
                 c_dim=3,
                 bnDecay=0.9,
                 epsilon=1e-5,
                 gp_scale=10.0,
                 y_dim=1):
        super(ACGAN, self).__init__(sess,
                                   model_name,
                                   result_path,
                                   random_dim,
                                   img_size,
                                   c_dim,
                                   bnDecay,
                                   epsilon)
        self.gp_scale = gp_scale
        self.y_dim = y_dim
    
    def build_model(self):
        image_dims = [self.img_size, self.img_size, self.c_dim]
        self.input_images = tf.placeholder(tf.float32, shape=[None]+image_dims, name='input_images')
        self.input_labels = tf.placeholder(tf.float32, shape=[None, self.y_dim], name='input_labels')
        self.z_random = tf.placeholder(tf.float32, shape=[None, self.random_dim], name='latent_vec')
        self.bn_train = tf.placeholder('bool')
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        
        ## batch normalization
        self.d_bn0 = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='d_bn0')
        self.d_bn1 = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='d_bn1')
        self.d_bn2 = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='d_bn2')
        self.d_bn3 = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='d_bn3')
        self.g_bn0 = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='g_bn0')
        self.g_bn1 = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='g_bn1')
        self.g_bn2 = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='g_bn2')
        self.g_bn3 = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='g_bn3')
        
        ## training data operations
        self.image_fake = self.generator(self.z_random, self.input_labels, self.bn_train)
        self.d_gan_logit_real, self.d_aux_logit_real = self.discriminator(self.input_images, self.bn_train)
        self.d_gan_logit_fake, self.d_aux_logit_fake = self.discriminator(self.image_fake, self.bn_train, reuse=True)
        self.image_sample = self.generator(self.z_random, self.input_labels, bn_train=False, reuse=True)
        
        ## loss
        ### Original GAN loss
        self.loss_d_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_gan_logit_real,
                                                                                  labels=tf.ones_like(self.d_gan_logit_real)))
        self.loss_d_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_gan_logit_fake,
                                                                                  labels=tf.zeros_like(self.d_gan_logit_fake)))
        self.loss_d = self.loss_d_real + self.loss_d_fake
        self.loss_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_gan_logit_fake,
                                                                             labels=tf.ones_like(self.d_gan_logit_fake)))
        ### AUX loss
        self.loss_aux_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_aux_logit_real,
                                                                                    labels=self.input_labels))
        self.loss_aux_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_aux_logit_fake,
                                                                                    labels=self.input_labels))
        self.loss_aux = self.loss_aux_real + self.loss_aux_fake
        ### GAN + AUX
        self.loss_all_d = self.loss_d + self.loss_aux
        self.loss_all_g = self.loss_g + self.loss_aux
        
        ## training operations
        train_vars = tf.trainable_variables()
        self.vars_d = [var for var in train_vars if 'discriminator' in var.name]
        self.vars_g = [var for var in train_vars if 'generator' in var.name]
        self.train_op_d = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                 beta1=0.5).minimize(self.loss_all_d,
                                                                     var_list=self.vars_d)
        self.train_op_g = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                 beta1=0.5).minimize(self.loss_all_g,
                                                                     var_list=self.vars_g)
        
        ## Create model saver (keep all checkpoint!)
        self.saver = tf.train.Saver(max_to_keep = None)
    
    def discriminator(self, input_images, bn_train, reuse = False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
            h0 = lrelu(self.d_bn0(conv2d(input_images, output_dim=32, name='h0'), train=bn_train), leak=0.1) ## [-1, 32, 32, 32]
            h1 = lrelu(self.d_bn1(conv2d(h0, output_dim=64, name='h1'), train=bn_train), leak=0.1) ## [-1, 16, 16, 64]
            h2 = lrelu(self.d_bn2(conv2d(h1, output_dim=128, name='h2'), train=bn_train), leak=0.1) ## [-1, 8, 8, 128]
            h3 = lrelu(self.d_bn3(conv2d(h2, output_dim=256, name='h3'), train=bn_train), leak=0.1) ## [-1, 4, 4, 256]
            h4_gan = linear(tf.reshape(h3, [-1, 4096]), 1, 'h4_gan')
            h4_aux = linear(tf.reshape(h3, [-1, 4096]), self.y_dim, 'h4_aux')
            return h4_gan, h4_aux
    
    def generator(self, z_random, input_labels, bn_train, reuse = False):
        with tf.variable_scope("generator") as scope:
            if reuse:
                scope.reuse_variables()
            bsize = tf.shape(z_random)[0]
            inputs = tf.concat([z_random, input_labels], axis=1)
            h0 = tf.reshape(linear(inputs, 4096, 'h0'), [-1, 4, 4, 256])
            h0 = tf.nn.relu(self.g_bn0(h0, train=bn_train))
            h1 = deconv2d(h0, [bsize, 8, 8, 128], name='h1')
            h1 = tf.nn.relu(self.g_bn1(h1, train=bn_train))
            h2 = deconv2d(h1, [bsize, 16, 16, 64], name='h2')
            h2 = tf.nn.relu(self.g_bn2(h2, train=bn_train))
            h3 = deconv2d(h2, [bsize, 32, 32, 32], name='h3')
            h3 = tf.nn.relu(self.g_bn3(h3, train=bn_train))
            h4 = deconv2d(h3, [bsize, 64, 64, 3], name='h4')
            return (tf.tanh(h4)/2. + 0.5)
    
    def train(self,
              init_from=None,
              train_path='/data/put_data/cclin/ntu/dlcv2018/hw4/hw4_data/train',
              test_path='/data/put_data/cclin/ntu/dlcv2018/hw4/hw4_data/test',
              train_period_d=1,
              train_period_g=2,
              nEpochs=200,
              bsize=32,
              learning_rate_start=1e-3,
              patience=10,
              attr_name='Smiling'):
        ## create a dedicated folder for this model
        if os.path.exists(os.path.join(self.result_path, self.model_name)):
            print('WARNING: the folder "{}" already exists!'.format(os.path.join(self.result_path, self.model_name)))
        else:
            os.makedirs(os.path.join(self.result_path, self.model_name))
            os.makedirs(os.path.join(self.result_path, self.model_name, 'samples'))
            os.makedirs(os.path.join(self.result_path, self.model_name, 'models'))
        
        ## data list (it is allowed to use both training and testing images)
        data_list = np.sort(glob.glob(os.path.join(train_path, '*.png')))
        if test_path is not None:
            data_list = np.concatenate((data_list, np.sort(glob.glob(os.path.join(test_path, '*.png')))), axis=0)
        nBatches = int(np.ceil(len(data_list) / bsize))
        idx_all = [i for i in range(len(data_list))]
        ### read attributes
        attr = pd.read_csv(os.path.join(os.path.dirname(train_path), 'train.csv'))
        attr_names = list(attr.columns.values)[1:]
        if test_path is not None:
            attr = pd.concat([attr, pd.read_csv(os.path.join(os.path.dirname(test_path), 'test.csv'))])
        label_list = np.array(attr[attr_name])
        
        ## initialization
        initOp = tf.global_variables_initializer()
        self.sess.run(initOp)
        
        ## main training loop
        loss_aux_real_epoch = []
        loss_aux_fake_epoch = []
        accuracy_real_epoch = []
        accuracy_fake_epoch = []
        loss_aux_real_list = []
        loss_aux_fake_list = []
        accuracy_real_list = []
        accuracy_fake_list = []
        best_vae_loss = 0
        stopping_step = 0
        for epoch in range(1, (nEpochs+1)):
            loss_aux_real_batch = []
            loss_aux_fake_batch = []
            accuracy_real_batch = []
            accuracy_fake_batch = []
            idx = 0
            np.random.shuffle(idx_all)
            while idx < nBatches:
                #### update D once
                #for _ in range(train_period_d):
                if True:
                    batch_idxs = idx_all[idx*bsize:(idx+1)*bsize]
                    batch_files = data_list[batch_idxs]
                    if len(batch_files) == 0:
                        idx = nBatches
                        break
                    batch = [get_image(batch_file) for batch_file in batch_files]
                    batch_images = np.array(batch).astype(np.float32)
                    batch_labels = np.array(label_list[batch_idxs]).astype(np.float32)
                    batch_labels = np.expand_dims(batch_labels, axis=1)
                    batch_z_random = np.random.uniform(-1, 1, [batch_images.shape[0], self.random_dim]).astype(np.float32)
                    temp_results = self.sess.run([self.train_op_d,
                                                  self.loss_aux_real,
                                                  self.loss_aux_fake,
                                                  self.d_aux_logit_real,
                                                  self.d_aux_logit_fake],
                                                 feed_dict={self.input_images: batch_images,
                                                            self.input_labels: batch_labels,
                                                            self.z_random: batch_z_random,
                                                            self.bn_train: True,
                                                            self.learning_rate: learning_rate_start})
                    loss_aux_real_batch.append(temp_results[1])
                    loss_aux_fake_batch.append(temp_results[2])
                    accuracy_real = accuracy_score(temp_results[3] >= 0, batch_labels == 1.0)
                    accuracy_real_batch.append(accuracy_real)
                    accuracy_fake = accuracy_score(temp_results[4] >= 0, batch_labels == 1.0)
                    accuracy_fake_batch.append(accuracy_fake)
                    idx += 1
                #### update G once
                #for _ in range(train_period_g):
                if True:
                    #batch_z_random = np.random.uniform(-1, 1, [bsize, self.random_dim]).astype(np.float32)
                    self.sess.run(self.train_op_g,
                                  feed_dict={self.input_images: batch_images,
                                             self.input_labels: batch_labels,
                                             self.z_random: batch_z_random,
                                             self.bn_train: True,
                                             self.learning_rate: learning_rate_start})
                    
            ### record aux loss and accuracy for each iteration
            loss_aux_real_list.extend(loss_aux_real_batch)
            loss_aux_fake_list.extend(loss_aux_fake_batch)
            accuracy_real_list.extend(accuracy_real_batch)
            accuracy_fake_list.extend(accuracy_fake_batch)
            ### record (averaged) aux loss and accuracy for each epoch
            loss_aux_real_epoch.append(np.mean(loss_aux_real_batch))
            loss_aux_fake_epoch.append(np.mean(loss_aux_fake_batch))
            accuracy_real_epoch.append(np.mean(accuracy_real_batch))
            accuracy_fake_epoch.append(np.mean(accuracy_fake_batch))
            
            print('Epoch: %d, loss_aux_d: %f, loss_aux_g: %f, accuracy_real: %f, accuracy_fake: %f' % \
                  (epoch,
                   np.mean(loss_aux_real_batch),
                   np.mean(loss_aux_fake_batch),
                   np.mean(accuracy_real_batch),
                   np.mean(accuracy_fake_batch)))
            
            ### save model and run inference for every 10 epochs
            if epoch % 10 == 0:
                #### plot 10 randomly generated images with opposite attrbutes
                batch_z_random = np.random.uniform(-1, 1, [10, self.random_dim]).astype(np.float32)
                batch_z_random = np.concatenate((batch_z_random, batch_z_random), axis=0)
                batch_labels = np.array([np.repeat((0, 1), 10)]).astype(np.float32).reshape((20, 1))
                
                samples = self.sess.run(self.image_sample, feed_dict={self.z_random: batch_z_random,
                                                                      self.input_labels: batch_labels,
                                                                      self.bn_train: False})
                fig = self.plot(samples, 2, 10)
                plt.savefig(os.path.join(self.result_path, self.model_name, 'samples', '{}.png'.format(str(epoch).zfill(3))), 
                            bbox_inches='tight')
                plt.close(fig)
                #### save model only if epoch >= 100 (more stable)
                if epoch >= 100:
                    self.saver.save(self.sess,
                                    os.path.join(self.result_path, self.model_name, 'models', self.model_name + '.model'),
                                    global_step=epoch)
            
        return [loss_aux_real_list, loss_aux_fake_list, loss_aux_real_epoch, loss_aux_fake_epoch,
                accuracy_real_list, accuracy_fake_list, accuracy_real_epoch, accuracy_fake_epoch]
    
    def inference(self,
                  gen_from=None,
                  gen_from_ckpt=None,
                  out_path=None,
                  bsize=32,
                  set_seed=1002):
        ## create output folder
        if gen_from is None:
            gen_from = os.path.join(self.result_path, self.model_name, 'models')
        if out_path is not None:
            if os.path.exists(out_path):
                print('WARNING: the output path "{}" already exists!'.format(out_path))
            else:
                os.makedirs(out_path)
        else:
            out_path = os.path.join(self.result_path, self.model_name)
        
        ## load previous model if possible
        could_load, checkpoint_counter = self.load(gen_from, gen_from_ckpt)
        if could_load:
            print(" [*] Load SUCCESS")
            ### set seed to generate identical figures for the report
            np.random.seed(set_seed)
            
            ### fig3_2.jpg: learning curve
            #### Assume that 'results.npy' (containing losses vs. training iterations)
            #### is saved in the same directory as the model checkpoint
            results = np.load(os.path.join(gen_from, 'results.npy'))
            nBatch = len(results[0]) // len(results[2])
            fig, ax = plt.subplots(1,2, figsize=(16,6))
            ax[0].plot(range(len(results[0])), results[0],
                       label='Real', color='royalblue', alpha = 0.2)
            ax[0].plot(range(len(results[1])), results[1],
                       label='Fake', color='tomato', alpha = 0.2)
            ax[0].plot([i*nBatch for i in range(len(results[2]))], results[2],
                       label='Real (per epoch)', color='royalblue', linewidth=3.0, alpha = 0.7)
            ax[0].plot([i*nBatch for i in range(len(results[3]))], results[3],
                       label='Fake (per epoch)', color='tomato', linewidth=3.0, alpha = 0.7)
            ax[0].set_xlabel('Training iterations')
            ax[0].set_title('Training Loss of Attribute Classification', fontsize=20)
            ax[0].legend(loc="upper right", fontsize=16)
            ax[1].plot(range(len(results[4])), results[4],
                       label='Real', color='royalblue', alpha = 0.2)
            ax[1].plot(range(len(results[5])), results[5],
                       label='Fake', color='tomato', alpha = 0.2)
            ax[1].plot([i*nBatch for i in range(len(results[6]))], results[6],
                       label='Real (per epoch)', color='royalblue', linewidth=3.0, alpha = 0.7)
            ax[1].plot([i*nBatch for i in range(len(results[7]))], results[7],
                       label='Fake (per epoch)', color='tomato', linewidth=3.0, alpha = 0.7)
            ax[1].set_xlabel('Training iterations')
            ax[1].set_title('Accuracy of Discriminator', fontsize=20)
            ax[1].legend(loc="lower right", fontsize=16)
            plt.savefig(os.path.join(out_path, 'fig3_2.jpg'))
            plt.close(fig)
            
            #### fig3_3.jpg: plot 10 randomly generated images with opposite attrbutes
            batch_z_random = np.random.uniform(-1, 1, [10, self.random_dim]).astype(np.float32)
            batch_z_random = np.concatenate((batch_z_random, batch_z_random), axis=0)
            batch_labels = np.array([np.repeat((0, 1), 10)]).astype(np.float32).reshape((20, 1))

            samples = self.sess.run(self.image_sample, feed_dict={self.z_random: batch_z_random,
                                                                  self.input_labels: batch_labels,
                                                                  self.bn_train: False})
            fig = self.plot(samples, 2, 10)
            plt.savefig(os.path.join(out_path, 'fig3_3.jpg'), bbox_inches='tight')
            plt.close(fig)
    
class INFOGAN(GAN):
    def __init__(self,
                 sess,
                 model_name='INFOGAN',
                 result_path='/data/put_data/cclin/ntu/dlcv2018/hw4/results',
                 random_dim=100,
                 img_size=64,
                 c_dim=3,
                 bnDecay=0.9,
                 epsilon=1e-5,
                 gp_scale=10.0,
                 # default: 10 ten-dimensional categorical codes
                 n_cat=10,
                 cat_dim=10,
                 # default: 2 continuous codes
                 cont_dim=2):
        super(INFOGAN, self).__init__(sess,
                                   model_name,
                                   result_path,
                                   random_dim,
                                   img_size,
                                   c_dim,
                                   bnDecay,
                                   epsilon)
        self.gp_scale = gp_scale
        self.n_cat = n_cat
        self.cat_dim = cat_dim
        self.cont_dim = cont_dim
    
    def build_model(self):
        image_dims = [self.img_size, self.img_size, self.c_dim]
        self.input_images = tf.placeholder(tf.float32, shape=[None]+image_dims, name='input_images')
        self.input_labels = tf.placeholder(tf.float32, shape=[None, self.n_cat * self.cat_dim + self.cont_dim], name='input_labels')
        self.z_random = tf.placeholder(tf.float32, shape=[None, self.random_dim], name='latent_vec')
        self.bn_train = tf.placeholder('bool')
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        
        ## batch normalization
        self.d_bn0 = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='d_bn0')
        self.d_bn1 = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='d_bn1')
        self.d_bn2 = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='d_bn2')
        self.d_bn3 = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='d_bn3')
        self.g_bn0 = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='g_bn0')
        self.g_bn1 = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='g_bn1')
        self.g_bn2 = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='g_bn2')
        self.g_bn3 = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='g_bn3')
        self.q_bn0 = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='q_bn0')
        
        ## training data operations
        self.image_fake = self.generator(self.z_random, self.input_labels, self.bn_train)
        self.d_logit_real, _ = self.discriminator(self.input_images, self.bn_train)
        self.d_logit_fake, self.label_given_img = self.discriminator(self.image_fake, self.bn_train, reuse=True)
        self.image_sample = self.generator(self.z_random, self.input_labels, bn_train=False, reuse=True)
        
        ## loss
        ### D
        self.loss_d_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_logit_real,
                                                                                  labels=tf.ones_like(self.d_logit_real)))
        self.loss_d_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_logit_fake,
                                                                                  labels=tf.zeros_like(self.d_logit_fake)))
        self.loss_d = self.loss_d_real + self.loss_d_fake
        ### G
        self.loss_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_logit_fake,
                                                                             labels=tf.ones_like(self.d_logit_fake)))
        ### Q
        self.cond_entropy = tf.reduce_mean(-tf.reduce_sum(tf.log(self.label_given_img + 1e-8) * self.input_labels, axis=1))
        self.entropy = tf.reduce_mean(-tf.reduce_sum(tf.log(self.input_labels + 1e-8) * self.input_labels, axis=1))
        self.loss_q = self.cond_entropy + self.entropy
        
        ## training operations
        train_vars = tf.trainable_variables()
        self.vars_d = [var for var in train_vars if ('discriminator' in var.name and (not 'Q' in var.name))]
        self.vars_g = [var for var in train_vars if 'generator' in var.name]
        self.vars_q = [var for var in train_vars if 'Q' in var.name]
        ### debug:
        #print(len(self.vars_d))
        #print(len(self.vars_g))
        #print(len(self.vars_q))
        
        self.train_op_d = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                 beta1=0.5).minimize(self.loss_d,
                                                                     var_list=self.vars_d)
        self.train_op_g = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                 beta1=0.5).minimize(self.loss_g,
                                                                     var_list=self.vars_g)
        self.train_op_q = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                 beta1=0.5).minimize(self.loss_q,
                                                                     var_list=self.vars_g+self.vars_q)
        
        ## Create model saver (keep all checkpoint!)
        self.saver = tf.train.Saver(max_to_keep = None)
    
    def discriminator(self, input_images, bn_train, reuse = False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
            h0 = lrelu(self.d_bn0(conv2d(input_images, output_dim=32, name='h0'), train=bn_train), leak=0.1) ## [-1, 32, 32, 32]
            h1 = lrelu(self.d_bn1(conv2d(h0, output_dim=64, name='h1'), train=bn_train), leak=0.1) ## [-1, 16, 16, 64]
            h2 = lrelu(self.d_bn2(conv2d(h1, output_dim=128, name='h2'), train=bn_train), leak=0.1) ## [-1, 8, 8, 128]
            h3 = lrelu(self.d_bn3(conv2d(h2, output_dim=256, name='h3'), train=bn_train), leak=0.1) ## [-1, 4, 4, 256]
            h4 = linear(tf.reshape(h3, [-1, 4096]), 1, 'h4')
            ### construct Q
            h0_Q = lrelu(self.q_bn0(linear(tf.reshape(h3, [-1, 4096]), 128, 'h0_Q')))
            if not self.n_cat * self.cat_dim == 0:
                h1_Q_cat = tf.nn.softmax(linear(tf.reshape(h0_Q, [-1, 128]), self.cat_dim, 'h1_Q_cat_0'))
                for n in range(1, self.n_cat):
                    h1_Q_cat = tf.concat([h1_Q_cat,
                                          tf.nn.softmax(linear(tf.reshape(h0_Q, [-1, 128]), self.cat_dim, 'h1_Q_cat_'+str(n)))],
                                         axis=1)
            if not self.cont_dim == 0:
                h1_Q_cont = tf.nn.sigmoid(linear(tf.reshape(h0_Q, [-1, 128]), self.cont_dim, 'h1_Q_cont'))
            if self.n_cat == 0:
                Q_out = h1_Q_cont
            elif self.cont_dim == 0:
                Q_out = h1_Q_cat
            else: ### there must be some code
                Q_out = tf.concat([h1_Q_cat, h1_Q_cont], axis=1)
            return h4, Q_out
    
    def generator(self, z_random, input_labels, bn_train, reuse = False):
        with tf.variable_scope("generator") as scope:
            if reuse:
                scope.reuse_variables()
            bsize = tf.shape(z_random)[0]
            inputs = tf.concat([z_random, input_labels], axis=1)
            h0 = tf.reshape(linear(inputs, 4096, 'h0'), [-1, 4, 4, 256])
            h0 = tf.nn.relu(self.g_bn0(h0, train=bn_train))
            h1 = deconv2d(h0, [bsize, 8, 8, 128], name='h1')
            h1 = tf.nn.relu(self.g_bn1(h1, train=bn_train))
            h2 = deconv2d(h1, [bsize, 16, 16, 64], name='h2')
            h2 = tf.nn.relu(self.g_bn2(h2, train=bn_train))
            h3 = deconv2d(h2, [bsize, 32, 32, 32], name='h3')
            h3 = tf.nn.relu(self.g_bn3(h3, train=bn_train))
            h4 = deconv2d(h3, [bsize, 64, 64, 3], name='h4')
            return (tf.tanh(h4)/2. + 0.5)
    
    def train(self,
              init_from=None,
              train_path='/data/put_data/cclin/ntu/dlcv2018/hw4/hw4_data/train',
              test_path='/data/put_data/cclin/ntu/dlcv2018/hw4/hw4_data/test',
              train_period_d=1,
              train_period_g=2,
              nEpochs=200,
              bsize=32,
              learning_rate_start=1e-3,
              patience=10):
        ## create a dedicated folder for this model
        if os.path.exists(os.path.join(self.result_path, self.model_name)):
            print('WARNING: the folder "{}" already exists!'.format(os.path.join(self.result_path, self.model_name)))
        else:
            os.makedirs(os.path.join(self.result_path, self.model_name))
            os.makedirs(os.path.join(self.result_path, self.model_name, 'samples'))
            os.makedirs(os.path.join(self.result_path, self.model_name, 'models'))
        
        ## data list (it is allowed to use both training and testing images)
        data_list = np.sort(glob.glob(os.path.join(train_path, '*.png')))
        if test_path is not None:
            data_list = np.concatenate((data_list, np.sort(glob.glob(os.path.join(test_path, '*.png')))), axis=0)
        nBatches = int(np.ceil(len(data_list) / bsize))
        idx_all = [i for i in range(len(data_list))]
        
        ## initialization
        initOp = tf.global_variables_initializer()
        self.sess.run(initOp)
        
        ## main training loop
        loss_d_list = []
        loss_g_list = []
        loss_q_list = []
        best_vae_loss = 0
        stopping_step = 0
        for epoch in range(1, (nEpochs+1)):
            loss_d_batch = []
            loss_g_batch = []
            loss_q_batch = []
            idx = 0
            np.random.shuffle(idx_all)
            while idx < nBatches:
                #### update D once
                #for _ in range(train_period_d):
                if True:
                    batch_files = data_list[idx*bsize:(idx+1)*bsize]
                    if len(batch_files) == 0:
                        idx = nBatches
                        break
                    batch = [get_image(batch_file) for batch_file in batch_files]
                    batch_images = np.array(batch).astype(np.float32)
                    batch_labels_cat = np.random.multinomial(1,
                                                             [1/self.cat_dim] * self.cat_dim,
                                                             size=batch_images.shape[0]).astype(np.float32)
                    for n in range(self.n_cat-1):
                        batch_labels_cat_temp = np.random.multinomial(1,
                                                                      [1/self.cat_dim] * self.cat_dim,
                                                                      size=batch_images.shape[0]).astype(np.float32)
                        batch_labels_cat = np.concatenate((batch_labels_cat, batch_labels_cat_temp), axis=1)
                    batch_labels_cont = np.random.uniform(0, 1, [batch_images.shape[0], self.cont_dim]).astype(np.float32)
                    batch_labels = np.concatenate((batch_labels_cat, batch_labels_cont), axis=1)
                    batch_z_random = np.random.uniform(-1, 1, [batch_images.shape[0], self.random_dim]).astype(np.float32)
                    _, loss_d = self.sess.run([self.train_op_d, self.loss_d], feed_dict={self.input_images: batch_images,
                                                                                         self.input_labels: batch_labels,
                                                                                         self.z_random: batch_z_random,
                                                                                         self.bn_train: True,
                                                                                         self.learning_rate: learning_rate_start})
                    loss_d_batch.append(loss_d)
                    idx += 1
                #### update G once
                #for _ in range(train_period_g):
                if True:
                    #batch_z_random = np.random.uniform(-1, 1, [bsize, self.random_dim]).astype(np.float32)
                    _, loss_g = self.sess.run([self.train_op_g, self.loss_g], feed_dict={self.input_labels: batch_labels,
                                                                                         self.z_random: batch_z_random,
                                                                                         self.bn_train: True,
                                                                                         self.learning_rate: learning_rate_start})
                    loss_g_batch.append(loss_g)
                    _, loss_q = self.sess.run([self.train_op_q, self.loss_q], feed_dict={self.input_labels: batch_labels,
                                                                                         self.z_random: batch_z_random,
                                                                                         self.bn_train: True,
                                                                                         self.learning_rate: learning_rate_start})
                    loss_q_batch.append(loss_q)
                    
            ### record aux loss and accuracy for each iteration
            loss_d_list.extend(loss_d_batch)
            loss_g_list.extend(loss_g_batch)
            loss_q_list.extend(loss_q_batch)
            
            print('Epoch: %d, loss_d: %f, loss_g: %f, loss_q: %f' % \
                  (epoch,
                   np.mean(loss_d_batch),
                   np.mean(loss_g_batch),
                   np.mean(loss_q_batch)))
            
            ### save model and run inference for every 10 epochs
            if epoch % 10 == 0:
                #### fix the random noise throughout this inference
                batch_z_random = np.random.uniform(-1, 1, [4, self.random_dim]).astype(np.float32) ##### shape: (4, self.random_dim)
                batch_z_random = np.repeat(batch_z_random, 10, axis = 0) ##### shape: (40, self.random_dim)
                #### Manipulate categorical codes (fix all continuous codes)
                if not self.n_cat * self.cat_dim == 0:                    
                    #### create trivial labels of the form: [00000000010000000001...0000000001] with shape self.n_cat * self.cat_dim
                    trivial_labels = np.array([i % self.cat_dim == 0 for i in range(1,self.n_cat*self.cat_dim+1)]).astype(np.float32)
                    #### enumerate a specific categorical code or loop all 10 categories
                    #target_cat = 0
                    for target_cat in range(self.n_cat):
                        for pos in range(self.cat_dim):
                            ##### [NOTE]: don't just write batch_labels_temp = trivial_labels
                            batch_labels_temp = [i for i in trivial_labels]
                            batch_labels_temp[target_cat*self.cat_dim:(target_cat+1)*self.cat_dim] = \
                                np.array([i % self.cat_dim == pos for i in range(self.cat_dim)]).astype(np.float32)
                            if pos == 0:
                                batch_enum = batch_labels_temp
                            else:
                                batch_enum = np.vstack((batch_enum, batch_labels_temp))
                            ##### batch_enum.shape: (self.cat_dim, self.n_cat * self.cat_dim)
                            ##### For example, n_cat = 5, cat_dim = 3, target_cat = 1:
                            #[[0. 0. 1. 1. 0. 0. 0. 0. 1. 0. 0. 1. 0. 0. 1.]
                            # [0. 0. 1. 0. 1. 0. 0. 0. 1. 0. 0. 1. 0. 0. 1.]
                            # [0. 0. 1. 0. 0. 1. 0. 0. 1. 0. 0. 1. 0. 0. 1.]]
                        ##### append fixed continuous codes
                        if not self.cont_dim == 0:
                            batch_constant = np.repeat(np.expand_dims(np.repeat(0.5, self.cont_dim), axis=0), self.cat_dim, axis=0)
                            batch_enum = np.hstack((batch_enum, batch_constant))
                            ##### batch_enum.shape: (self.cat_dim, self.n_cat * self.cat_dim + self.cont_dim)
                            ##### For example, n_cat = 5, cat_dim = 3, target_cat = 1, cont_dim = 2:
                            #[[0. 0. 1. 1. 0. 0. 0. 0. 1. 0. 0. 1. 0. 0. 1. 0.5 0.5]
                            # [0. 0. 1. 0. 1. 0. 0. 0. 1. 0. 0. 1. 0. 0. 1. 0.5 0.5]
                            # [0. 0. 1. 0. 0. 1. 0. 0. 1. 0. 0. 1. 0. 0. 1. 0.5 0.5]]
                        ##### repeat 4 time to make shape: (4 * self.cat_dim, self.n_cat * self.cat_dim + self.cont_dim)
                        batch_labels = np.concatenate((batch_enum, batch_enum, batch_enum, batch_enum), axis=0)
                        ##### run session
                        samples = self.sess.run(self.image_sample, feed_dict={self.z_random: batch_z_random,
                                                                              self.input_labels: batch_labels,
                                                                              self.bn_train: False})
                        fig = self.plot(samples, 4, self.cat_dim)
                        plt.savefig(os.path.join(self.result_path, self.model_name,
                                                 'samples', '{}_cat{}.png'.format(str(epoch).zfill(3), str(target_cat).zfill(2))), 
                                    bbox_inches='tight')
                        plt.close(fig)
                ### Manipulate continuous codes (fix all categorical codes)
                if not self.cont_dim == 0:
                    batch_linspace = np.linspace(-0.5, 1.5, num=10)
                    batch_constant = np.repeat(0.5, 10)
                    for target_dim in range(self.cont_dim):
                        batch_labels = np.empty((10, self.cont_dim)).astype(np.float32)
                        for i in range(self.cont_dim):
                            if i == target_dim:
                                batch_labels[:,i] = batch_linspace
                            else:
                                batch_labels[:,i] = batch_constant
                            #### batch_labels.shape: (10, self.cont_dim)
                            ##### For example, cont_dim = 2, target_dim = 0:
                            #[[-0.5         0.5       ]
                            # [-0.27777778  0.5       ]
                            # [-0.05555556  0.5       ]
                            # [ 0.16666667  0.5       ]
                            # [ 0.38888889  0.5       ]
                            # [ 0.61111111  0.5       ]
                            # [ 0.83333333  0.5       ]
                            # [ 1.05555556  0.5       ]
                            # [ 1.27777778  0.5       ]
                            # [ 1.5         0.5       ]]
                        ##### append fixed categorical codes
                        if not self.n_cat * self.cat_dim == 0:
                            trivial_labels = np.array([i % self.cat_dim == 0 \
                                                       for i in range(1,self.n_cat*self.cat_dim+1)]).astype(np.float32)
                            trivial_labels_10 = np.repeat(np.expand_dims(trivial_labels, axis=0), 10, axis=0)
                            batch_labels = np.hstack((batch_labels, trivial_labels_10))
                        ##### repeat 4 time to make shape: (40, self.cont_dim * self.cat_dim + self.cont_dim)
                        batch_labels = np.concatenate((batch_labels, batch_labels, batch_labels, batch_labels), axis=0)
                        ##### run session
                        samples = self.sess.run(self.image_sample, feed_dict={self.z_random: batch_z_random,
                                                                              self.input_labels: batch_labels,
                                                                              self.bn_train: False})
                        fig = self.plot(samples, 4, 10)
                        plt.savefig(os.path.join(self.result_path, self.model_name,
                                                 'samples', '{}_cont{}.png'.format(str(epoch).zfill(3), str(target_dim).zfill(2))), 
                                    bbox_inches='tight')
                        plt.close(fig)
                #### save model only if epoch >= 100 (more stable)
                if epoch >= 100:
                    self.saver.save(self.sess,
                                    os.path.join(self.result_path, self.model_name, 'models', self.model_name + '.model'),
                                    global_step=epoch)
            
        return [loss_d_list, loss_g_list, loss_q_list]
    
    def inference(self,
                  gen_from=None,
                  gen_from_ckpt=None,
                  out_path=None,
                  bsize=32,
                  set_seed=1002):
        ## create output folder
        if gen_from is None:
            gen_from = os.path.join(self.result_path, self.model_name, 'models')
        if out_path is not None:
            if os.path.exists(out_path):
                print('WARNING: the output path "{}" already exists!'.format(out_path))
            else:
                os.makedirs(out_path)
        else:
            out_path = os.path.join(self.result_path, self.model_name)
        
        ## load previous model if possible
        could_load, checkpoint_counter = self.load(gen_from, gen_from_ckpt)
        if could_load:
            print(" [*] Load SUCCESS")
            ### set seed to generate identical figures for the report
            np.random.seed(set_seed)
            
            ### fig4_1.jpg: learning curve
            #### Assume that 'results.npy' (containing losses vs. training iterations)
            #### is saved in the same directory as the model checkpoint
            results = np.load(os.path.join(gen_from, 'results.npy'))
            
            #### enumerate each categorical code one at a time
            n_row = 10
            ##### (1) fix the pure random vector
            batch_z_random = np.random.uniform(-1, 1, [n_row, self.random_dim]).astype(np.float32)
            batch_z_random = np.repeat(batch_z_random, self.cat_dim, axis = 0)
            for cat in range(self.n_cat):
                batch_labels = np.empty([n_row, self.cat_dim, self.n_cat * self.cat_dim])
                for rep in range(n_row):
                    ##### randomly create the prefixed and postfixed categorical codes (repeat n_row times)
                    vec_pre = np.random.multinomial(1, [1/self.cat_dim] * self.cat_dim, size=cat).astype(np.float32)
                    vec_pre = np.reshape(vec_pre, [-1])
                    vec_post = np.random.multinomial(1, [1/self.cat_dim] * self.cat_dim, size=self.n_cat - cat - 1).astype(np.float32)
                    vec_post = np.reshape(vec_post, [-1])
                    ##### create one-hot vector for each digit from 0 to self.cat_dim-1, and concatenate with pre and post
                    for num in range(self.cat_dim):
                        vec = np.array([i%self.cat_dim==num for i in range(self.cat_dim)]).astype(np.float32)
                        vec = np.concatenate((vec_pre, vec, vec_post))
                        vec = np.expand_dims(vec, axis = 0)
                        batch_labels[rep, num, :] = vec
                batch_labels = np.reshape(batch_labels, [-1, self.n_cat * self.cat_dim])
                ##### plot
                samples = self.sess.run(self.image_sample, feed_dict={self.z_random: batch_z_random,
                                                                      self.input_labels: batch_labels,
                                                                      self.bn_train: False})
                fig = self.plot(samples, n_row, self.cat_dim)
                plt.savefig(os.path.join(out_path, 'enum_cat{}.jpg'.format(str(cat).zfill(2))), bbox_inches='tight')
                plt.close(fig)
            
            #### alternative: use a single random noise vector and enumerate 2 categorical vectors
            ####              to see the 100 variations at a time
            if False:
                for sample_idx in range(5):
                    ##### (1) fix the pure random vector
                    batch_z_random = np.random.uniform(-1, 1, [1, self.random_dim]).astype(np.float32)
                    batch_z_random = np.repeat(batch_z_random, 100, axis = 0)
                    ##### (2) random generate the rest categorical codes
                    res_n_cat = self.n_cat - 2
                    batch_labels_cat_temp = np.random.multinomial(1, [1/self.cat_dim] * self.cat_dim, size=res_n_cat).astype(np.float32)
                    batch_labels_cat_temp = np.reshape(batch_labels_cat_temp, [-1])
                    ##### (3) create one-hot vector for each of the first/last 2 digits from 0 to 99
                    #####     (making 2 ten-dimensional vectors for each number)
                    batch_labels_first2 = np.empty([100, 50])
                    batch_labels_last2 = np.empty([100, 50])
                    for num in range(100):
                        d0 = num % 10
                        d1 = num // 10
                        vec = np.array([i%10==d0 for i in range(10)]+[i%10==d1 for i in range(10)]).astype(np.float32)
                        vec_first2 = np.concatenate((vec, batch_labels_cat_temp))
                        vec_first2 = np.expand_dims(vec_first2, axis = 0)
                        batch_labels_first2[num,:] = vec_first2
                        vec_last2 = np.concatenate((batch_labels_cat_temp, vec))
                        vec_last2 = np.expand_dims(vec_last2, axis = 0)
                        batch_labels_last2[num,:] = vec_last2
                    ##### first 2 digits
                    samples = self.sess.run(self.image_sample, feed_dict={self.z_random: batch_z_random,
                                                                          self.input_labels: batch_labels_first2,
                                                                          self.bn_train: False})
                    fig = self.plot(samples, 10, 10)
                    plt.savefig(os.path.join(out_path, 'grid_{}_10by10_of_first2.jpg'.format(str(sample_idx).zfill(2))))
                    plt.close(fig)
                    ##### last 2 digits
                    samples = self.sess.run(self.image_sample, feed_dict={self.z_random: batch_z_random,
                                                                          self.input_labels: batch_labels_last2,
                                                                          self.bn_train: False})
                    fig = self.plot(samples, 10, 10)
                    plt.savefig(os.path.join(out_path, 'grid_{}_10by10_of_last2.jpg'.format(str(sample_idx).zfill(2))))
                    plt.close(fig)


