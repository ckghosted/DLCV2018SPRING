import os, re, time, glob
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import batch_norm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tqdm

import skimage
import skimage.transform
import skimage.io

from ops import *
from utils import *

class VAE(object):
    def __init__(self,
                 sess,
                 model_name='VAE',
                 result_path='/data/put_data/cclin/ntu/dlcv2018/hw4/results',
                 input_height=64,
                 input_width=64,
                 c_dim=3,
                 output_height=64,
                 output_width=64,
                 latent_dim=512,
                 bnDecay=0.99,
                 lambda_kl=1e-5,
                 l2scale=0.001):
        self.sess = sess
        self.result_path = result_path
        self.model_name = model_name
        self.input_height = input_height
        self.input_width = input_width
        self.c_dim = c_dim
        self.output_height = output_height
        self.output_width = output_width
        self.latent_dim = latent_dim
        self.bnDecay = bnDecay
        self.epsilon = 1 - bnDecay
        self.l2scale = l2scale
        
        self.lambda_kl = lambda_kl
        
    
    # Build the VAE
    def build_model(self):
        image_dims = [self.input_height, self.input_width, self.c_dim]
        self.input_images = tf.placeholder(tf.float32, shape=[None]+image_dims, name='input_images')
        self.input_images_ = tf.placeholder(tf.float32, shape=[None]+image_dims, name='input_images_')
        self.z_random = tf.placeholder(tf.float32, shape=[None, self.latent_dim], name='latent_vec')
        self.bn_train = tf.placeholder('bool')
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        
        ## batch normalization
        self.enc_bn0 = batch_norm(name='enc_bn0')
        self.enc_bn1 = batch_norm(name='enc_bn1')
        self.enc_bn2 = batch_norm(name='enc_bn2')
        self.enc_bn3 = batch_norm(name='enc_bn3')
        self.enc_bn4_mu = batch_norm(name='enc_bn4_mu')
        self.enc_bn4_logvar = batch_norm(name='enc_bn4_logvar')
        self.dec_bn0 = batch_norm(name='dec_bn0')
        self.dec_bn1 = batch_norm(name='dec_bn1')
        self.dec_bn2 = batch_norm(name='dec_bn2')
        self.dec_bn3 = batch_norm(name='dec_bn3')
        
        ## training data operations
        self.z_mu, self.z_logvar = self.encoder(self.input_images, self.bn_train)
        self.z_sample = self.sample_z(self.z_mu, self.z_logvar)
        self.recon_images = self.decoder(self.z_sample, self.bn_train)
        
        ## Sampling from random z
        self.sample_images = self.decoder(self.z_random, self.bn_train, reuse=True)
        
        ## E[log P(X|z)]
        self.recon_loss = tf.reduce_mean((self.input_images - self.recon_images)**2, [1, 2, 3])
        ## D_KL(Q(z|X) || P(z)); calculate in closed form as both dist. are Gaussian
        self.kl_loss = 0.5 * tf.reduce_mean(tf.exp(self.z_logvar) + self.z_mu**2 - 1. - self.z_logvar, 1)
        ## VAE loss
        self.vae_loss = tf.reduce_mean(self.recon_loss + self.lambda_kl * self.kl_loss)
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.vae_loss)
        ## Create model saver (keep the best 3 checkpoint)
        self.saver = tf.train.Saver(max_to_keep = 3)
       
    def encoder(self, input_images, bn_train):
        with tf.variable_scope("encoder") as scope:
            h0 = lrelu(self.enc_bn0(conv2d(input_images, output_dim=32, name='h0'), train=bn_train)) ## [-1, 32, 32, 32]
            h1 = lrelu(self.enc_bn1(conv2d(h0, output_dim=64, name='h1'), train=bn_train)) ## [-1, 16, 16, 64]
            h2 = lrelu(self.enc_bn2(conv2d(h1, output_dim=128, name='h2'), train=bn_train)) ## [-1, 8, 8, 128]
            h3 = lrelu(self.enc_bn3(conv2d(h2, output_dim=256, name='h3'), train=bn_train)) ## [-1, 4, 4, 256]
            z_mu = linear(tf.reshape(h3, [-1, 4096]), self.latent_dim, 'z_mu')
            z_logvar = linear(tf.reshape(h3, [-1, 4096]), self.latent_dim, 'z_logvar')
            return z_mu, z_logvar
    
    def sample_z(self, mu, log_var):
        with tf.variable_scope("sample_z") as scope:
            eps = tf.random_normal(shape=tf.shape(mu))
            return mu + tf.exp(log_var / 2) * eps
    
    def decoder(self, z_sample, bn_train, reuse = False):
        with tf.variable_scope("decoder") as scope:
            if reuse:
                scope.reuse_variables()
            bsize = tf.shape(z_sample)[0]
            h0 = tf.reshape(linear(z_sample, 4096, 'h0'), [-1, 4, 4, 256])
            h0 = lrelu(self.dec_bn0(h0, train=bn_train))
            h1 = deconv2d(h0, [bsize, 8, 8, 128], name='h1')
            h1 = lrelu(self.dec_bn1(h1, train=bn_train))
            h2 = deconv2d(h1, [bsize, 16, 16, 64], name='h2')
            h2 = lrelu(self.dec_bn2(h2, train=bn_train))
            h3 = deconv2d(h2, [bsize, 32, 32, 32], name='h3')
            h3 = lrelu(self.dec_bn3(h3, train=bn_train))
            h4 = deconv2d(h3, [bsize, 64, 64, 3], name='h4')
            return (tf.tanh(h4)/2. + 0.5)
    
    def train(self,
              init_from=None,
              train_path='/data/put_data/cclin/ntu/dlcv2018/hw4/hw4_data/train',
              test_path='/data/put_data/cclin/ntu/dlcv2018/hw4/hw4_data/test',
              nEpochs=200,
              bsize=32,
              learning_rate_start=1e-5,
              patience=3):
        ## create a dedicated folder for this model
        if os.path.exists(os.path.join(self.result_path, self.model_name)):
            print('WARNING: the folder "{}" already exists!'.format(os.path.join(self.result_path, self.model_name)))
        else:
            os.makedirs(os.path.join(self.result_path, self.model_name))
            os.makedirs(os.path.join(self.result_path, self.model_name, 'samples'))
            os.makedirs(os.path.join(self.result_path, self.model_name, 'recons'))
            os.makedirs(os.path.join(self.result_path, self.model_name, 'models'))
        
        ## data list
        train_list = glob.glob(os.path.join(train_path, '*.png'))
        nBatches = int(np.ceil(len(train_list) / bsize))
        test_list = glob.glob(os.path.join(test_path, '*.png'))
        nBatches_test = int(np.ceil(len(test_list) / bsize))
        
        ## initialization
        initOp = tf.global_variables_initializer()
        self.sess.run(initOp)
        
        ## main training loop
        mse_train = []
        kld_train = []
        mse_test = []
        kld_test = []
        best_mse_test = 0
        stopping_step = 0
        for epoch in range(1, (nEpochs+1)):
            mse_train_batch = []
            kld_train_batch = []
            for idx in tqdm.tqdm(range(nBatches)):
                batch_files = train_list[idx*bsize:(idx+1)*bsize]
                batch = [get_image(batch_file,
                                   input_height=self.input_height,
                                   input_width=self.input_width,
                                   resize_height=self.output_height,
                                   resize_width=self.output_width) for batch_file in batch_files]
                batch_images = np.array(batch).astype(np.float32)
                _, mse, kld = self.sess.run([self.train_op, self.recon_loss, self.kl_loss],
                                            feed_dict={self.input_images: batch_images,
                                                       self.bn_train: True,
                                                       self.learning_rate: learning_rate_start})
                mse_train_batch.append(np.mean(mse))
                kld_train_batch.append(np.mean(kld))
            mse_train.append(np.mean(mse_train_batch))
            kld_train.append(np.mean(kld_train_batch))
            
            ### compute testing loss
            mse_test_batch = []
            kld_test_batch = []
            for idx in tqdm.tqdm(range(nBatches_test)):
                batch_files = test_list[idx*bsize:(idx+1)*bsize]
                batch = [get_image(batch_file,
                                   input_height=self.input_height,
                                   input_width=self.input_width,
                                   resize_height=self.output_height,
                                   resize_width=self.output_width) for batch_file in batch_files]
                batch_images = np.array(batch).astype(np.float32)
                mse, kld = self.sess.run([self.recon_loss, self.kl_loss],
                                         feed_dict={self.input_images: batch_images,
                                                    self.bn_train: False})
                mse_test_batch.append(np.mean(mse))
                kld_test_batch.append(np.mean(kld))
            mse_test.append(np.mean(mse_test_batch))
            kld_test.append(np.mean(kld_test_batch))
            print('Epoch: %d, train mse: %f, train kld: %f, test mse: %f, test kld: %f' % \
                  (epoch, np.mean(mse_train_batch), np.mean(kld_train_batch), np.mean(mse_test_batch), np.mean(kld_test_batch)))
            
            ### save model if improvement
            if epoch == 1:
                best_mse_test = np.mean(mse_test_batch)
            else:
                if np.mean(mse_test_batch) < best_mse_test:
                    best_mse_test = np.mean(mse_test_batch)
                    self.saver.save(self.sess,
                                    os.path.join(self.result_path, self.model_name, 'models', self.model_name + '.model'),
                                    global_step=epoch)
                    stopping_step = 0
                else:
                    stopping_step += 1
                if stopping_step >= patience:
                    print('stopping_step >= patience (%d), stop training' % patience)
                    break
            
            ### inference during training process
            if epoch % 5 == 0:
                #### reconstruct 10 testing images
                batch_files = np.sort(glob.glob(os.path.join(test_path, '4000*.png')))
                batch = [get_image(batch_file,
                                   input_height=self.input_height,
                                   input_width=self.input_width,
                                   resize_height=self.output_height,
                                   resize_width=self.output_width) for batch_file in batch_files]
                batch_images = np.array(batch).astype(np.float32)
                recons = self.sess.run(self.recon_images, feed_dict={self.input_images: batch_images,
                                                                      self.bn_train: False})
                fig = self.plot(np.concatenate((batch_images, recons), axis=0), 2, 10)
                plt.savefig(os.path.join(self.result_path, self.model_name, 'recons', '{}.png'.format(str(epoch).zfill(3))), 
                            bbox_inches='tight')
                plt.close(fig)
                
                #### produce 32 random images
                samples = self.sess.run(self.sample_images, feed_dict={self.z_random: np.random.randn(32, self.latent_dim),
                                                                       self.bn_train: False})
                fig = self.plot(samples, 4, 8)
                plt.savefig(os.path.join(self.result_path, self.model_name, 'samples', '{}.png'.format(str(epoch).zfill(3))), 
                            bbox_inches='tight')
                plt.close(fig)
                
                #### visualization
                z_mu_all = []
                for idx in tqdm.tqdm(range(nBatches_test)):
                    batch_files = test_list[idx*bsize:(idx+1)*bsize]
                    batch = [get_image(batch_file,
                                       input_height=self.input_height,
                                       input_width=self.input_width,
                                       resize_height=self.output_height,
                                       resize_width=self.output_width) for batch_file in batch_files]
                    batch_images = np.array(batch).astype(np.float32)
                    z_mu = self.sess.run(self.z_mu,
                                         feed_dict={self.input_images: batch_images,
                                                    self.bn_train: False})
                    z_mu_all.extend(z_mu)
                z_mu_all = np.array(z_mu_all).astype(np.float32)
                print(z_mu_all.shape)
                ## TODO: t-SNE and visualization
                
        return [mse_train, kld_train, mse_test, kld_test]
    
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






