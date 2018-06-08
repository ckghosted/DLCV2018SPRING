import os, re, time, glob
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib import rnn

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import skimage
import skimage.transform
import skimage.io

import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score

from ops import *
from utils import *
from reader import *

VGG_MEAN = [103.939, 116.779, 123.68]

class EXTRACTOR(object):
    def __init__(self,
                 sess,
                 model_name='EXTRACTOR',
                 result_path='/data/put_data/cclin/ntu/dlcv2018/hw5/results',
                 img_size_h=240,
                 img_size_w=320,
                 c_dim=3):
        self.sess = sess
        self.model_name = model_name
        self.result_path = result_path
        self.img_size_h = img_size_h
        self.img_size_w = img_size_w
        self.c_dim = c_dim
    
    ## Build VGG16 as the feature extractor
    def build_vgg16(self,
                    vgg16_npy_path='/data/put_data/cclin/ntu/dlcv2018/hw3/vgg16.npy'):
        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        print("vgg16.npy loaded")
        
        image_dims = [self.img_size_h, self.img_size_w, self.c_dim]
        self.images = tf.placeholder(tf.float32, shape=[None]+image_dims, name='images')
        
        print("RGB to BGR")
        # rgb_scaled = rgb * 255.0
        ### Input layer: convert RGB to BGR and subtract pixels mean
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=self.images)
        self.bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        
        print("build model started")
        ### ------------------------------------------------------------
        ### VGG conv+pooling part. Note that only max_pool(.) will halve
        ### the feature map size (both H and W) by a factor of 2, while
        ### all conv_layer(.) keep the same feature map size.
        ### ------------------------------------------------------------
        ### Layer 1
        self.conv1_1 = self.conv_layer(self.bgr, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')
        ### Layer 2
        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')
        ### Layer 3
        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self.max_pool(self.conv3_3, 'pool3')
        ### Layer 4
        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self.max_pool(self.conv4_3, 'pool4')
        ### Layer 5
        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.pool5 = self.max_pool(self.conv5_3, 'pool5')
    
    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
    
    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)
            #out = batch_norm(bias,
            #                 decay=self.bnDecay,
            #                 epsilon=self.epsilon,
            #                 scale=True,
            #                 is_training=self.bn_train,
            #                 updates_collections=None)
            #relu = tf.nn.relu(out)
            relu = tf.nn.relu(bias)
            # relu = utils.leaky_relu(bias, alpha=0.2, name='lrelu'+name[-3:])
            return relu
    
    def get_conv_filter(self, name):
        var=tf.Variable(self.data_dict[name][0], name="filter_"+name)
        return var
    
    def get_bias(self, name):
        return tf.Variable(self.data_dict[name][1], name="biases_"+name)
    
    def extract(self,
                video_path,
                label_file,
                downsample_factor=12,
                rescale_factor=1):
        video_list = getVideoList(label_file)
        n_video = len(video_list['Video_name'])
        
        ### initialization
        initOp = tf.global_variables_initializer()
        self.sess.run(initOp)
        
        features_all = []
        for i in range(len(video_list['Video_name'])):
            feature_video = []
            frames = readShortVideo(video_path,
                                    video_list['Video_category'][i],
                                    video_list['Video_name'][i],
                                    downsample_factor,
                                    rescale_factor)
            for f in range(frames.shape[0]):
                feature_frame = self.sess.run(self.pool5, feed_dict={self.images: np.expand_dims(frames[f], axis=0)})
                feature_video.append(feature_frame)
            feature_video = np.array(feature_video).astype(np.float32) #### shape: (frames.shape[0], 1, 8, 10, 512)
            feature_video = np.squeeze(feature_video, axis=1) #### shape: (frames.shape[0], 8, 10, 512)
            features_all.append(feature_video.reshape([frames.shape[0], -1])) #### append shape: (frames.shape[0], 8*10*512)
        return features_all

class DNN(object):
    def __init__(self,
                 sess,
                 model_name='DNN',
                 result_path='/data/put_data/cclin/ntu/dlcv2018/hw5/results',
                 n_class=11,
                 bnDecay=0.9,
                 epsilon=1e-5):
        self.sess = sess
        self.model_name = model_name
        self.result_path = result_path
        self.n_class = n_class
        self.bnDecay = bnDecay
        self.epsilon = epsilon
    
    ## Build the discriminative model
    def build_model(self, feature_dim=40960):
        self.features = tf.placeholder(tf.float32, shape=[None, feature_dim], name='features')
        self.labels = tf.placeholder(tf.int32, shape=[None, self.n_class], name='labels')
        self.bn_train = tf.placeholder('bool')
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        
        ## batch normalization
        self.bn1 = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='bn1')
        self.bn2 = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='bn2')
        
        ## fully-connected layers
        h1 = linear(self.features, 2048, 'h1')
        h1 = tf.nn.relu(self.bn1(h1, train=self.bn_train))
        h2 = linear(h1, 512, 'h2')
        h2 = tf.nn.relu(self.bn2(h2, train=self.bn_train))
        self.logits = linear(h2, self.n_class, 'logits')
        
        ## loss function
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels,
                                                                           logits=self.logits,
                                                                           name='loss'))
        
        ## training operation
        self.t_vars = tf.trainable_variables()
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5).minimize(self.loss, var_list=self.t_vars)
        
        ## Create model saver (keep the best 3 checkpoint)
        self.saver = tf.train.Saver(max_to_keep = 3)
    
    def train(self,
              feature_path_train='/data/put_data/cclin/ntu/dlcv2018/hw5/cnn_features_train.npy',
              label_file_train='/data/put_data/cclin/ntu/dlcv2018/hw5/HW5_data/TrimmedVideos/label/gt_train.csv',
              feature_path_valid='/data/put_data/cclin/ntu/dlcv2018/hw5/cnn_features_valid.npy',
              label_file_valid='/data/put_data/cclin/ntu/dlcv2018/hw5/HW5_data/TrimmedVideos/label/gt_valid.csv',
              bsize=64,
              learning_rate=2e-4,
              num_epoch=200,
              patience=10):
        ## create a dedicated folder for this model
        if os.path.exists(os.path.join(self.result_path, self.model_name)):
            print('WARNING: the folder "{}" already exists!'.format(os.path.join(self.result_path, self.model_name)))
        else:
            os.makedirs(os.path.join(self.result_path, self.model_name))
            os.makedirs(os.path.join(self.result_path, self.model_name, 'models'))
        
        ## load training features and take average over all frames of a video
        features_train = [np.mean(f, axis=0) for f in np.load(feature_path_train)]
        nBatches = int(np.ceil(len(features_train) / bsize))
        ## load training labels (and make them one-hot vectors)
        video_list_train = getVideoList(label_file_train)
        labels_train = [int(s) for s in video_list_train['Action_labels']]
        labels_train_vec = np.eye(self.n_class)[labels_train]
        ## load validation features and take average over all frames of a video
        features_valid = [np.mean(f, axis=0) for f in np.load(feature_path_valid)]
        nBatches_valid = int(np.ceil(len(features_valid) / bsize))
        ## load validation labels (and make them one-hot vectors)
        video_list_valid = getVideoList(label_file_valid)
        labels_valid = [int(s) for s in video_list_valid['Action_labels']]
        labels_valid_vec = np.eye(self.n_class)[labels_valid]
        
        ## initialization
        initOp = tf.global_variables_initializer()
        self.sess.run(initOp)
        
        ## main training loop
        loss_train = []
        loss_valid = []
        acc_train = []
        acc_valid = []
        best_loss = 0
        stopping_step = 0
        for epoch in range(1, (num_epoch+1)):
            loss_train_batch = []
            loss_valid_batch = []
            acc_train_batch = []
            acc_valid_batch = []
            for idx in range(nBatches):
                batch_features = features_train[idx*bsize:(idx+1)*bsize]
                batch_labels = labels_train_vec[idx*bsize:(idx+1)*bsize]
                _, loss, logits = self.sess.run([self.train_op, self.loss, self.logits],
                                                feed_dict={self.features: batch_features,
                                                           self.labels: batch_labels,
                                                           self.bn_train: True,
                                                           self.learning_rate: learning_rate})
                loss_train_batch.append(loss)
                y_true = np.argmax(batch_labels, axis=1)
                y_pred = np.argmax(logits, axis=1)
                acc_train_batch.append(accuracy_score(y_true, y_pred))
            ### compute validation loss
            for idx in range(nBatches_valid):
                batch_features = features_valid[idx*bsize:(idx+1)*bsize]
                batch_labels = labels_valid_vec[idx*bsize:(idx+1)*bsize]
                loss, logits = self.sess.run([self.loss, self.logits],
                                             feed_dict={self.features: batch_features,
                                                        self.labels: batch_labels,
                                                        self.bn_train: False})
                loss_valid_batch.append(loss)
                y_true = np.argmax(batch_labels, axis=1)
                y_pred = np.argmax(logits, axis=1)
                acc_valid_batch.append(accuracy_score(y_true, y_pred))
            ### record training loss for each epoch (instead of each iteration)
            loss_train.append(np.mean(loss_train_batch))
            loss_valid.append(np.mean(loss_valid_batch))
            acc_train.append(np.mean(acc_train_batch))
            acc_valid.append(np.mean(acc_valid_batch))
            print('Epoch: %d, train loss: %f, valid loss: %f, train accuracy: %f, valid accuracy: %f' % \
                  (epoch, np.mean(loss_train_batch), np.mean(loss_valid_batch), np.mean(acc_train_batch), np.mean(acc_valid_batch)))
            
            ### save model if improvement, stop if reach patience
            current_loss = np.mean(loss_valid_batch)
            if epoch == 1:
                best_loss = current_loss
            else:
                if current_loss < best_loss:
                    best_loss = current_loss
                    self.saver.save(self.sess,
                                    os.path.join(self.result_path, self.model_name, 'models', self.model_name + '.model'),
                                    global_step=epoch)
                    stopping_step = 0
                else:
                    stopping_step += 1
                print('stopping_step = %d' % stopping_step)
                if stopping_step >= patience:
                    print('stopping_step >= patience (%d), stop training' % patience)
                    break
        return [loss_train, loss_valid, acc_train, acc_valid]
    
    def inference(self,
                  #feature_path='/data/put_data/cclin/ntu/dlcv2018/hw5/cnn_features_valid.npy',
                  cnn_features,
                  label_file='/data/put_data/cclin/ntu/dlcv2018/hw5/HW5_data/TrimmedVideos/label/gt_valid.csv',
                  gen_from=None,
                  gen_from_ckpt=None,
                  out_path=None,
                  bsize=64):
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
            
            ## load features and take average over all frames of a video
            #features = [np.mean(f, axis=0) for f in np.load(feature_path)]
            features = [np.mean(f, axis=0) for f in cnn_features]
            nBatches = int(np.ceil(len(features) / bsize))
            ## load labels (and make them one-hot vectors)
            video_list = getVideoList(label_file)
            labels = [int(s) for s in video_list['Action_labels']]
            labels_vec = np.eye(self.n_class)[labels]
            
            ### p1_valid.txt: make prediction
            loss_batch = []
            acc_batch = []
            y_pred_list = []
            for idx in range(nBatches):
                batch_features = features[idx*bsize:(idx+1)*bsize]
                batch_labels = labels_vec[idx*bsize:(idx+1)*bsize]
                loss, logits = self.sess.run([self.loss, self.logits],
                                             feed_dict={self.features: batch_features,
                                                        self.labels: batch_labels,
                                                        self.bn_train: False})
                loss_batch.append(loss)
                y_true = np.argmax(batch_labels, axis=1)
                y_pred = np.argmax(logits, axis=1)
                acc_batch.append(accuracy_score(y_true, y_pred))
                y_pred_list.extend(y_pred)
            print('valid loss: %f, valid accuracy: %f' % \
                  (np.mean(loss_batch), np.mean(acc_batch)))
            with open(os.path.join(out_path, 'p1_valid.txt'), 'w') as f:
                for y in y_pred_list:
                    f.write(str(y)+'\n')
    
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

class LSTM(object):
    def __init__(self,
                 sess,
                 model_name='LSTM',
                 result_path='/data/put_data/cclin/ntu/dlcv2018/hw5/results',
                 n_class=11,
                 max_seq_len=10,
                 h_dim=512,
                 bnDecay=0.9,
                 epsilon=1e-5):
        self.sess = sess
        self.model_name = model_name
        self.result_path = result_path
        self.n_class = n_class
        self.max_seq_len = max_seq_len
        self.h_dim = h_dim
        self.bnDecay = bnDecay
        self.epsilon = epsilon
    
    ## Build the discriminative model
    def build_model(self,
                    use_static_rnn=True,
                    feature_dim=40960):
        self.features = tf.placeholder(tf.float32, shape=[None, self.max_seq_len, feature_dim], name='features')
        self.labels = tf.placeholder(tf.int32, shape=[None, self.n_class], name='labels')
        self.seqlen = tf.placeholder(tf.int32, [None])
        self.bn_train = tf.placeholder('bool')
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        
        ## batch normalization
        self.bn1 = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='bn1')
        #self.bn2 = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='bn2')
        
        ## Approach 1: use static_rnn() with sequence_length to perform dynamic calculation
        if use_static_rnn:
            ### Unstack to get a list of 'max_seq_len' tensors of shape (batch_size, n_input)
            x = tf.unstack(self.features, self.max_seq_len, 1)
            ### (1) single layer:
            #lstm_cell = rnn.BasicLSTMCell(self.h_dim, forget_bias=1.0)
            #outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32, sequence_length=self.seqlen)
            ### (2) multiple layers:
            rnn_layers = [rnn.BasicLSTMCell(self.h_dim, forget_bias=1.0),
                          rnn.BasicLSTMCell(self.h_dim, forget_bias=1.0)]
            multi_rnn_cell = rnn.MultiRNNCell(rnn_layers)
            outputs, states = rnn.static_rnn(multi_rnn_cell, x, dtype=tf.float32, sequence_length=self.seqlen)
            ### In this case, 'outputs' will be a list of length self.max_seq_len,
            ### with each element being a tensor of shape (batch_size, self.h_dim).
            outputs = tf.stack(outputs) ### shape: (self.max_seq_len, batch_size, self.h_dim)
            outputs = tf.transpose(outputs, [1, 0, 2]) ### shape: (batch_size, self.max_seq_len, self.h_dim)
        ## Approach 2: dynamic_rnn()
        else:
            ### (1) single layer:
            #lstm_cell = rnn.BasicLSTMCell(self.h_dim, forget_bias=1.0)
            #outputs, states = tf.nn.dynamic_rnn(lstm_cell, self.features, dtype=tf.float32, sequence_length=self.seqlen)
            ### (2) multiple layers:
            rnn_layers = [rnn.BasicLSTMCell(self.h_dim, forget_bias=1.0),
                          rnn.BasicLSTMCell(self.h_dim, forget_bias=1.0)]
            multi_rnn_cell = rnn.MultiRNNCell(rnn_layers)
            outputs, states = tf.nn.dynamic_rnn(multi_rnn_cell, self.features, dtype=tf.float32, sequence_length=self.seqlen)
            ### In this case, 'outputs' will be a tensor of shape (batch_size, self.max_seq_len, self.h_dim).
        
        ## Hack to build the indexing and retrieve the right output:
        batch_size = tf.shape(outputs)[0]
        ## start indices for each sample
        index = tf.range(0, batch_size) * self.max_seq_len + (self.seqlen - 1)
        ## indexing
        outputs = tf.gather(tf.reshape(outputs, [-1, self.h_dim]), index)
        
        ## take the last output of the rnn inner loop
        h1 = linear(outputs, self.h_dim/2, 'h1')
        h1 = tf.nn.relu(self.bn1(h1, train=self.bn_train))
        #h2 = linear(h1, 256, 'h2')
        #h2 = tf.nn.relu(self.bn2(h2, train=self.bn_train))
        self.logits = linear(h1, self.n_class, 'logits')
        
        ## loss function
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels))
        
        ## training operation
        self.t_vars = tf.trainable_variables()
        print(self.t_vars)
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5).minimize(self.loss, var_list=self.t_vars)
        
        ## Create model saver (keep the best 3 checkpoint)
        self.saver = tf.train.Saver(max_to_keep = 3)
    
    def train(self,
              feature_path_train='/data/put_data/cclin/ntu/dlcv2018/hw5/cnn_features_train.npy',
              label_file_train='/data/put_data/cclin/ntu/dlcv2018/hw5/HW5_data/TrimmedVideos/label/gt_train.csv',
              feature_path_valid='/data/put_data/cclin/ntu/dlcv2018/hw5/cnn_features_valid.npy',
              label_file_valid='/data/put_data/cclin/ntu/dlcv2018/hw5/HW5_data/TrimmedVideos/label/gt_valid.csv',
              bsize=32,
              learning_rate=2e-4,
              num_epoch=200,
              patience=10):
        ## create a dedicated folder for this model
        if os.path.exists(os.path.join(self.result_path, self.model_name)):
            print('WARNING: the folder "{}" already exists!'.format(os.path.join(self.result_path, self.model_name)))
        else:
            os.makedirs(os.path.join(self.result_path, self.model_name))
            os.makedirs(os.path.join(self.result_path, self.model_name, 'models'))
        
        ## load training features
        features_train = np.load(feature_path_train)
        nBatches = int(np.ceil(len(features_train) / bsize))
        ## down-sampling to make sure all videos have number of frames <= self.max_seq_len
        features_train_new = []
        for v_idx in range(len(features_train)):
            n_frames = features_train[v_idx].shape[0]
            step = (n_frames - 1) // self.max_seq_len + 1
            features_train_new.append(features_train[v_idx][np.arange(0,n_frames,step),:])
        ## load training labels (and make them one-hot vectors)
        video_list_train = getVideoList(label_file_train)
        labels_train = [int(s) for s in video_list_train['Action_labels']]
        labels_train_vec = np.eye(self.n_class)[labels_train]
        ## load validation features
        features_valid = np.load(feature_path_valid)
        nBatches_valid = int(np.ceil(len(features_valid) / bsize))
        ## down-sampling to make sure all videos have number of frames <= self.max_seq_len
        features_valid_new = []
        for v_idx in range(len(features_valid)):
            n_frames = features_valid[v_idx].shape[0]
            step = (n_frames - 1) // self.max_seq_len + 1
            features_valid_new.append(features_valid[v_idx][np.arange(0,n_frames,step),:])
        ## load validation labels (and make them one-hot vectors)
        video_list_valid = getVideoList(label_file_valid)
        labels_valid = [int(s) for s in video_list_valid['Action_labels']]
        labels_valid_vec = np.eye(self.n_class)[labels_valid]
        
        ## initialization
        initOp = tf.global_variables_initializer()
        self.sess.run(initOp)
        
        ## main training loop
        loss_train = []
        loss_valid = []
        acc_train = []
        acc_valid = []
        best_loss = 0
        stopping_step = 0
        
        #print('start training')
        
        for epoch in range(1, (num_epoch+1)):
            loss_train_batch = []
            loss_valid_batch = []
            acc_train_batch = []
            acc_valid_batch = []
            for idx in range(nBatches-1):
                #print(idx)
                batch_features = features_train_new[idx*bsize:(idx+1)*bsize]
                ### padding 0
                batch_features_pad = np.array([np.vstack((f, np.zeros((self.max_seq_len - f.shape[0], f.shape[1])))) \
                                               if f.shape[0] < self.max_seq_len \
                                               else f[0:self.max_seq_len] \
                                               for f in batch_features])
                #print(batch_features_pad.shape)
                batch_labels = labels_train_vec[idx*bsize:(idx+1)*bsize]
                #print(batch_labels)
                batch_seqlen = np.array([np.min((f.shape[0], self.max_seq_len)) for f in batch_features], dtype=np.int32)
                #print(batch_seqlen)
                _, loss, logits = self.sess.run([self.train_op, self.loss, self.logits],
                                                feed_dict={self.features: batch_features_pad,
                                                           self.labels: batch_labels,
                                                           self.seqlen: batch_seqlen,
                                                           self.bn_train: True,
                                                           self.learning_rate: learning_rate})
                loss_train_batch.append(loss)
                y_true = np.argmax(batch_labels, axis=1)
                y_pred = np.argmax(logits, axis=1)
                acc_train_batch.append(accuracy_score(y_true, y_pred))
            ### compute validation loss
            for idx in range(nBatches_valid):
                #print(idx)
                batch_features = features_valid_new[idx*bsize:(idx+1)*bsize]
                ### padding 0
                batch_features_pad = np.array([np.vstack((f, np.zeros((self.max_seq_len - f.shape[0], f.shape[1])))) \
                                               if f.shape[0] < self.max_seq_len \
                                               else f[0:self.max_seq_len] \
                                               for f in batch_features])
                #print(batch_features_pad.shape)
                batch_labels = labels_valid_vec[idx*bsize:(idx+1)*bsize]
                #print(batch_labels)
                batch_seqlen = np.array([np.min((f.shape[0], self.max_seq_len)) for f in batch_features], dtype=np.int32)
                #print(batch_seqlen)
                loss, logits = self.sess.run([self.loss, self.logits],
                                             feed_dict={self.features: batch_features_pad,
                                                        self.labels: batch_labels,
                                                        self.seqlen: batch_seqlen,
                                                        self.bn_train: False})
                loss_valid_batch.append(loss)
                y_true = np.argmax(batch_labels, axis=1)
                y_pred = np.argmax(logits, axis=1)
                acc_valid_batch.append(accuracy_score(y_true, y_pred))
            ### record training loss for each epoch (instead of each iteration)
            loss_train.append(np.mean(loss_train_batch))
            loss_valid.append(np.mean(loss_valid_batch))
            acc_train.append(np.mean(acc_train_batch))
            acc_valid.append(np.mean(acc_valid_batch))
            print('Epoch: %d, train loss: %f, valid loss: %f, train accuracy: %f, valid accuracy: %f' % \
                  (epoch, np.mean(loss_train_batch), np.mean(loss_valid_batch), np.mean(acc_train_batch), np.mean(acc_valid_batch)))
            
            ### save model if improvement, stop if reach patience
            current_loss = np.mean(loss_valid_batch)
            if epoch == 1:
                best_loss = current_loss
            else:
                if current_loss < best_loss:
                    best_loss = current_loss
                    self.saver.save(self.sess,
                                    os.path.join(self.result_path, self.model_name, 'models', self.model_name + '.model'),
                                    global_step=epoch)
                    stopping_step = 0
                else:
                    stopping_step += 1
                print('stopping_step = %d' % stopping_step)
                if stopping_step >= patience:
                    print('stopping_step >= patience (%d), stop training' % patience)
                    break
        return [loss_train, loss_valid, acc_train, acc_valid]
    
    def inference(self,
                  #feature_path='/data/put_data/cclin/ntu/dlcv2018/hw5/cnn_features_valid.npy',
                  cnn_features,
                  label_file='/data/put_data/cclin/ntu/dlcv2018/hw5/HW5_data/TrimmedVideos/label/gt_valid.csv',
                  gen_from=None,
                  gen_from_ckpt=None,
                  out_path=None,
                  bsize=64):
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
            
            ## load features and take average over all frames of a video
            #features = np.load(feature_path)
            features = cnn_features
            nBatches = int(np.ceil(len(features) / bsize))
            ## down-sampling to make sure all videos have number of frames <= self.max_seq_len
            features_new = []
            for v_idx in range(len(features)):
                n_frames = features[v_idx].shape[0]
                step = (n_frames - 1) // self.max_seq_len + 1
                features_new.append(features[v_idx][np.arange(0,n_frames,step),:])
            ## load labels (and make them one-hot vectors)
            video_list = getVideoList(label_file)
            labels = [int(s) for s in video_list['Action_labels']]
            labels_vec = np.eye(self.n_class)[labels]
            
            ### p2_result.txt: make prediction
            loss_batch = []
            acc_batch = []
            y_pred_list = []
            for idx in range(nBatches):
                batch_features = features_new[idx*bsize:(idx+1)*bsize]
                ### padding 0
                batch_features_pad = np.array([np.vstack((f, np.zeros((self.max_seq_len - f.shape[0], f.shape[1])))) \
                                               if f.shape[0] < self.max_seq_len \
                                               else f[0:self.max_seq_len] \
                                               for f in batch_features])
                batch_labels = labels_vec[idx*bsize:(idx+1)*bsize]
                batch_seqlen = np.array([np.min((f.shape[0], self.max_seq_len)) for f in batch_features], dtype=np.int32)
                loss, logits = self.sess.run([self.loss, self.logits],
                                             feed_dict={self.features: batch_features_pad,
                                                        self.labels: batch_labels,
                                                        self.seqlen: batch_seqlen,
                                                        self.bn_train: False})
                loss_batch.append(loss)
                y_true = np.argmax(batch_labels, axis=1)
                y_pred = np.argmax(logits, axis=1)
                acc_batch.append(accuracy_score(y_true, y_pred))
                y_pred_list.extend(y_pred)
            print('valid loss: %f, valid accuracy: %f' % \
                  (np.mean(loss_batch), np.mean(acc_batch)))
            with open(os.path.join(out_path, 'p2_result.txt'), 'w') as f:
                for y in y_pred_list:
                    f.write(str(y)+'\n')
    
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






