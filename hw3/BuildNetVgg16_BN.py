# Class which build the fully convolutional neural net

import inspect
import os, re
import TensorflowUtils as utils
import Data_Reader
import numpy as np
import tensorflow as tf
import tqdm
import skimage
import skimage.transform
import skimage.io
from tensorflow.contrib.layers import batch_norm

## Mean value of pixels in B, G, and R channels, respectively
VGG_MEAN = [103.939, 116.779, 123.68]

## Class for building the FCN neural network based on VGG16
class NET_VGG16_BN:
    def __init__(self,
                 sess,
                 vgg16_npy_path,
                 model_name='VGG16_FCN32s',
                 result_path='/data/put_data/cclin/ntu/dlcv2018/hw3/results',
                 n_class=7,
                 fcn_stride=32,
                 img_size=256,
                 bnDecay=0.99):
        # if vgg16_npy_path is None:
        #     path = inspect.getfile(BUILD_NET_VGG16)
        #     path = os.path.abspath(os.path.join(path, os.pardir))
        #     path = os.path.join(path, "vgg16.npy")
        #     vgg16_npy_path = path
        #
        #     print(path)
        #### Load weights of trained VGG16 for encoder
        self.sess = sess
        self.result_path = result_path
        self.model_name = model_name
        self.n_class = n_class
        self.fcn_stride = fcn_stride
        self.img_size = img_size
        
        #### batch normalization parameters
        self.bnDecay = bnDecay
        self.epsilon = 1 - bnDecay
        
        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        print("vgg16.npy loaded")
    
    ### Build the FCN and load weights for decoder
    def build_model(self):
        """
        load variable from npy to build the VGG
        :param rgb: rgb image [batch, height, width, 3] values 0-255
        """
        #### Sum of weights of all filters for weight decay loss
        self.SumWeights = tf.constant(0.0, name="SumFiltersWeights")
        self.image = tf.placeholder(tf.float32, shape=[None, self.img_size, self.img_size, 3], name="input_image")
        self.label_true = tf.placeholder(tf.int32, shape=[None, self.img_size, self.img_size, 1], name="label_true")
        # self.keep_prob = tf.placeholder(tf.float32, name="keep_probabilty")
        self.bn_train = tf.placeholder('bool')
        self.learning_rate = tf.placeholder(tf.float32, shape=[]) ##### for adaptive learning rate
        
        print("RGB to BGR")
        # rgb_scaled = rgb * 255.0
        #### Input layer: convert RGB to BGR and subtract pixels mean
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=self.image)
        self.bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        print("build model started")
        #### ------------------------------------------------------------
        #### VGG conv+pooling part. Note that only max_pool(.) will halve
        #### the feature map size (both H and W) by a factor of 2, while
        #### all conv_layer(.) keep the same feature map size.
        #### ------------------------------------------------------------
        #### Layer 1
        self.conv1_1 = self.conv_layer(self.bgr, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')
        #### Layer 2
        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')
        #### Layer 3
        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self.max_pool(self.conv3_3, 'pool3')
        #### Layer 4
        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self.max_pool(self.conv4_3, 'pool4')
        #### Layer 5
        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.pool5 = self.max_pool(self.conv5_3, 'pool5')
        #### ------------------------------------------------------------
        #### Replace Dense layers of original VGG by convolutional layers.
        #### Note that all feature maps keep the same size (H and W), only
        #### depths are modified (512 --> 4096 --> 4096 --> self.n_class).
        #### ------------------------------------------------------------
        #### FCN 1
        W6 = utils.weight_variable([3, 3, 512, 4096], name="W6")
        b6 = utils.bias_variable([4096], name="b6")
        self.conv6 = utils.conv2d_basic(self.pool5, W6, b6)
        ##### https://www.tensorflow.org/api_docs/python/tf/contrib/layers/batch_norm
        self.conv6_bn = batch_norm(self.conv6,
                                   decay=self.bnDecay,
                                   epsilon=self.epsilon,
                                   scale=True,
                                   is_training=self.bn_train,
                                   updates_collections=None)
        self.relu6 = tf.nn.relu(self.conv6_bn, name="relu6")
        # self.relu6 = utils.leaky_relu(self.conv6, alpha=0.2, name="relu6")
        # if FLAGS.debug: utils.add_activation_summary(relu6)
        # self.relu_dropout6 = tf.nn.dropout(self.relu6, keep_prob=self.keep_prob)
        #### FCN 2 (1X1 convloution)
        W7 = utils.weight_variable([1, 1, 4096, 4096], name="W7")
        b7 = utils.bias_variable([4096], name="b7")
        self.conv7 = utils.conv2d_basic(self.relu6, W7, b7)
        ##### https://www.tensorflow.org/api_docs/python/tf/contrib/layers/batch_norm
        self.conv7_bn = batch_norm(self.conv7,
                                   decay=self.bnDecay,
                                   epsilon=self.epsilon,
                                   scale=True,
                                   is_training=self.bn_train,
                                   updates_collections=None)
        self.relu7 = tf.nn.relu(self.conv7_bn, name="relu7")
        # self.relu7 = utils.leaky_relu(self.conv7, alpha=0.2, name="relu7")
        # if FLAGS.debug: utils.add_activation_summary(relu7)
        # self.relu_dropout7 = tf.nn.dropout(self.relu7, keep_prob=self.keep_prob)
        #### FCN 3 (1X1 convloution)
        W8 = utils.weight_variable([1, 1, 4096, self.n_class], name="W8")
        b8 = utils.bias_variable([self.n_class], name="b8")
        self.conv8 = utils.conv2d_basic(self.relu7, W8, b8)
        # self.relu8 = tf.nn.relu(self.conv8, name="relu8")
        # annotation_pred1 = tf.argmax(conv8, axis=3, name="prediction1")
        #### ------------------------------------------------------------
        #### Upsampling by deconvolutional layers. Note that strides = 2
        #### by default for conv2d_transpose_strided(.).
        #### Also note that the "filter" parameter of tf.nn.conv2d_transpose(.)
        #### takes format: [height, width, output_channels, in_channels],
        #### while it is: [height, width, input_channels, output_channels]
        #### for tf.nn.conv2d(.)
        #### ------------------------------------------------------------
        #### Ref: https://blog.csdn.net/mao_xiao_feng/article/details/71713358
        if self.fcn_stride == 32:
            #### 32x upsampling
            shape = tf.shape(self.image)
            ##### Ref: https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/voc-fcn32s/net.py
            ##### (kernel_size=64 if stride=32)
            W_t3 = utils.weight_variable([64, 64, self.n_class, self.n_class], name="W_t3")
            # b_t3 = utils.bias_variable([self.n_class], name="b_t3")
            self.label_logit = utils.conv2d_transpose_strided(self.conv8,
                                                              W_t3,
                                                              # b_t3,
                                                              None,
                                                              output_shape=[shape[0], shape[1], shape[2], self.n_class],
                                                              stride=32)
        else: # (self.fcn_stride == 8)
            #### fuse_1 = 2x conv8 + pool4
            deconv_depth1 = self.pool4.get_shape()[3].value
            W_t1 = utils.weight_variable([4, 4, deconv_depth1, self.n_class], name="W_t1")
            # b_t1 = utils.bias_variable([deconv_depth1], name="b_t1")
            self.conv_t1 = utils.conv2d_transpose_strided(self.conv8,
                                                          W_t1,
                                                          # b_t1,
                                                          None,
                                                          output_shape=tf.shape(self.pool4))
            self.fuse_1 = tf.add(self.conv_t1, self.pool4, name="fuse_1")
            #### fuse_2 = 2x fuse_1 + pool3 = 4x conv8 + 2x pool4 + pool3
            deconv_depth2 = self.pool3.get_shape()[3].value
            W_t2 = utils.weight_variable([4, 4, deconv_depth2, deconv_depth1], name="W_t2")
            # b_t2 = utils.bias_variable([deconv_depth2], name="b_t2")
            self.conv_t2 = utils.conv2d_transpose_strided(self.fuse_1,
                                                          W_t2,
                                                          # b_t2,
                                                          None,
                                                          output_shape=tf.shape(self.pool3))
            self.fuse_2 = tf.add(self.conv_t2, self.pool3, name="fuse_2")
            #### 8x upsampling
            shape = tf.shape(self.image)
            W_t3 = utils.weight_variable([16, 16, self.n_class, deconv_depth2], name="W_t3")
            # b_t3 = utils.bias_variable([self.n_class], name="b_t3")
            self.label_logit = utils.conv2d_transpose_strided(self.fuse_2,
                                                              W_t3,
                                                              # b_t3,
                                                              None,
                                                              output_shape=[shape[0], shape[1], shape[2], self.n_class],
                                                              stride=8)
        
        #### Transform probability vectors to label maps
        self.label_predict = tf.argmax(self.label_logit, axis=3, name="label_predict")

        print("FCN model built")
        
        #### Define trainable variables and loss function
        self.t_vars = tf.trainable_variables()
        #### WARNING: This op expects unscaled logits, since it performs a softmax on logits internally for efficiency.
        ####          Do not call this op with the output of softmax, as it will produce incorrect results.
        self.loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.squeeze(self.label_true, squeeze_dims=[3]),
                                                                                   logits=self.label_logit,
                                                                                   name="loss")))
        ### define training operations
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss, var_list=self.t_vars)
        #### Create model saver
        self.saver = tf.train.Saver(max_to_keep = 1) ##### keep all checkpoints!
    
    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
    
    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)
            out = batch_norm(bias,
                             decay=self.bnDecay,
                             epsilon=self.epsilon,
                             scale=True,
                             is_training=self.bn_train,
                             updates_collections=None)
            relu = tf.nn.relu(out)
            # relu = utils.leaky_relu(bias, alpha=0.2, name='lrelu'+name[-3:])
            return relu
    
    def conv_layer_NoRelu(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)
            return bias
    
    ### Build fully convolutional layer
    def fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            #### Fully connected layer
            #### Note that the '+' operation automatically broadcasts the biases
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc
    
    ### Get VGG filter
    def get_conv_filter(self, name):
        var=tf.Variable(self.data_dict[name][0], name="filter_"+name)
        self.SumWeights+=tf.nn.l2_loss(var)
        return var
    
    def get_bias(self, name):
        return tf.Variable(self.data_dict[name][1], name="biases_"+name)
    
    def get_fc_weight(self, name):
        return tf.Variable(self.data_dict[name][0], name="weights_"+name)
    
    def train(self,
              init_from=None,
              train_path='/data/put_data/cclin/ntu/dlcv2018/hw3/train',
              valid_path='/data/put_data/cclin/ntu/dlcv2018/hw3/validation',
              nEpochs=10,
              bsize=2,
              # keep_prob_train=0.5,
              learning_rate_start=1e-5,
              patience=2):
        ### create a dedicated folder for this model
        if os.path.exists(os.path.join(self.result_path, self.model_name)):
            print('WARNING: the folder "{}" already exists!'.format(os.path.join(self.result_path, self.model_name)))
        else:
            os.makedirs(os.path.join(self.result_path, self.model_name))
            os.makedirs(os.path.join(self.result_path, self.model_name, 'outputs'))
            os.makedirs(os.path.join(self.result_path, self.model_name, 'models'))
        ### create logger
        log_path_train = os.path.join(self.result_path, self.model_name, 'training_loss.txt')
        f = open(log_path_train, "w")
        f.write("epoch\tloss\tlearning rate\n")
        f.close()
        if valid_path is not None:
            log_path_valid = os.path.join(self.result_path, self.model_name, 'validation_loss.txt')
            f = open(log_path_valid, "w")
            f.write("epoch\tloss\tlearning rate\n")
            f.close()
        ### data generator
        data_gen_train = Data_Reader.Data_Reader(train_path, GTLabelDir=train_path, BatchSize=bsize, img_size=self.img_size)
        if valid_path is not None:
            data_gen_valid = Data_Reader.Data_Reader(valid_path, GTLabelDir=valid_path, BatchSize=bsize, img_size=self.img_size)
        
        ### initialization
        initOp = tf.global_variables_initializer()
        self.sess.run(initOp)
        
        ### load previous model if possible
        train_from_scratch = True
        epoch_counter = 0
        if init_from is not None:
            could_load, checkpoint_counter = self.load(init_from)
            if could_load:
                epoch_counter = checkpoint_counter
                train_from_scratch = False
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")
        else:
            print(" [@] train from scratch")
        
        ### the main loop
        learning_rate_used = learning_rate_start
        best_loss = 0
        stopping_step = 0
        global_patience_count = 0
        for epoch in range(nEpochs):
            loss_temp = []
            nBatches = np.int(np.ceil(data_gen_train.NumFiles/data_gen_train.BatchSize))
            # print("Training on " + str(data_gen_train.NumFiles) + " images")
            for i in tqdm.tqdm(range(nBatches)):
                batch_image, batch_label_true, _ = data_gen_train.ReadNextBatchClean()
                _, TLoss = self.sess.run([self.train_op, self.loss], feed_dict = {self.image: batch_image,
                                                                                  self.label_true: batch_label_true,
                                                                                  # self.keep_prob: keep_prob_train,
                                                                                  self.learning_rate: learning_rate_used,
                                                                                  self.bn_train: True})
                loss_temp.append(TLoss)
            #### counter for file names of saved models
            epoch_counter += 1
            #### record training loss for every epoch
            buf = str(epoch_counter) + '\t' + str(np.mean(loss_temp)) + '\t' + str(learning_rate_used)
            self.print2file(buf, log_path_train)
            #### validation
            if valid_path is not None:
                if epoch_counter == 1 or epoch_counter % 10 == 0 or (epoch_counter > 60 and best_loss == np.mean(loss_temp)):
                    out_path = os.path.join(self.result_path, self.model_name, 'out_'+str(epoch_counter))
                    os.makedirs(out_path)
                loss_temp = []
                nBatches = np.int(np.ceil(data_gen_valid.NumFiles/data_gen_valid.BatchSize))
                # print("Calculating validation on " + str(data_gen_valid.NumFiles) + " images")
                for i in tqdm.tqdm(range(nBatches)):
                    batch_image, batch_label_true, batch_mask_path = data_gen_valid.ReadNextBatchClean()
                    VLoss = self.sess.run(self.loss, feed_dict = {self.image: batch_image,
                                                                  self.label_true: batch_label_true,
                                                                  # self.keep_prob: 1.0,
                                                                  self.bn_train: False})
                    loss_temp.append(VLoss)
                    #### run inference for every 10 epochs
                    if epoch_counter == 1 or epoch_counter % 10 == 0 or (epoch_counter > 60 and best_loss == np.mean(loss_temp)):
                        labels_pred = self.sess.run(self.label_predict, feed_dict = {self.image: batch_image,
                                                                                     # self.keep_prob: 1.0,
                                                                                     self.bn_train: False})
                        ##### resize labels_pred into out_img_size (e.g., 256 --> 512)
                        labels_pred = np.repeat(np.repeat(labels_pred, 2, axis=1), 2, axis=2)
                        for j in range(len(batch_mask_path)):
                            mask = self.label_to_rgb(labels_pred[j])
                            skimage.io.imsave(os.path.join(out_path, batch_mask_path[j]), mask)
                buf = str(epoch_counter) + '\t' + str(np.mean(loss_temp)) + '\t' + str(learning_rate_used)
                self.print2file(buf, log_path_valid)
                print('epoch_counter: %d, np.mean(loss_temp) = %f, best_loss = %f, stopping_step = %d' % (epoch_counter, np.mean(loss_temp), best_loss, stopping_step))
                #### update learning rate if necessary
                if epoch == 0:
                    best_loss = np.mean(loss_temp)
                else:
                    if (best_loss - np.mean(loss_temp)) > 0.0001:
                        best_loss = np.mean(loss_temp)
                        stopping_step = 0
                        ##### save model whenever improvement
                        save_path = self.saver.save(self.sess,
                                                    os.path.join(self.result_path, self.model_name, 'models', self.model_name + '.model'),
                                                    global_step=epoch_counter)
                    else:
                        stopping_step += 1
                    if stopping_step >= patience:
                        global_patience_count += 1
                        if global_patience_count < 3:
                            print("================================================")
                            print("Update learning rate from %f to %f" % (learning_rate_used, learning_rate_used / 2))
                            print("================================================")
                            stopping_step = 0
                            learning_rate_used = learning_rate_used / 2
                        else:
                            print("================================================")
                            print("Update learning rate %d times, stop training" % global_patience_count)
                            print("================================================")
                            break
                
    
    def inference(self,
                  test_path='/data/put_data/cclin/ntu/dlcv2018/hw3/validation',
                  gen_from=None,
                  gen_from_ckpt=None,
                  out_path=None,
                  bsize=2,
                  out_img_size=512):
        ### create output folder
        if gen_from is None:
            gen_from = os.path.join(self.result_path, self.model_name, 'models')
        if out_path is not None:
            if os.path.exists(out_path):
                print('WARNING: the output path "{}" already exists!'.format(out_path))
            else:
                os.makedirs(out_path)
        else:
            out_path = os.path.join(gen_from, 'outputs')
        
        ### load previous model if possible
        could_load, checkpoint_counter = self.load(gen_from, gen_from_ckpt)
        if could_load:
            print(" [*] Load SUCCESS")
            #### GTLabelDir='' to read image and mask file names ('xxxx_mask.png')
            data_gen_test = Data_Reader.Data_Reader(test_path, GTLabelDir='', BatchSize=bsize, img_size=self.img_size)
            nBatches = np.int(np.ceil(data_gen_test.NumFiles/data_gen_test.BatchSize))
            enlarge_factor = out_img_size / self.img_size
            # print('enlarge_factor = %d' % enlarge_factor)
            for i in tqdm.tqdm(range(nBatches)):
                batch_image, batch_mask_path = data_gen_test.ReadNextBatchClean()
                labels_pred = self.sess.run(self.label_predict, feed_dict = {self.image: batch_image,
                                                                             # self.keep_prob: 1.0,
                                                                             self.bn_train: False})
                ##### debug: also save smaller masks (e.g., 256)
                # for j in range(len(batch_mask_path)):
                #     mask = self.label_to_rgb(labels_pred[j])
                #     if not os.path.exists(os.path.join(out_folder_path, 'small')):
                #         os.makedirs(os.path.join(out_folder_path, 'small'))
                #     skimage.io.imsave(os.path.join(out_folder_path, 'small', batch_mask_path[j][0:-8] + 'mask_.png'), mask)
                
                ##### resize labels_pred into out_img_size (e.g., 256 --> 512)
                labels_pred = np.repeat(np.repeat(labels_pred, enlarge_factor, axis=1), enlarge_factor, axis=2)
                for j in range(len(batch_mask_path)):
                    mask = self.label_to_rgb(labels_pred[j])
                    skimage.io.imsave(os.path.join(out_path, batch_mask_path[j]), mask)
        else:
            print(" [*] Failed to find a checkpoint")
    
    def label_to_rgb(self, input_label):
        rgb = np.empty((input_label.shape+(3,)), dtype=int)
        rgb[input_label == 0] = 3  # (Cyan: 011) Urban land
        rgb[input_label == 1] = 6  # (Yellow: 110) Agriculture land
        rgb[input_label == 2] = 5  # (Purple: 101) Rangeland
        rgb[input_label == 3] = 2  # (Green: 010) Forest land
        rgb[input_label == 4] = 1  # (Blue: 001) Water
        rgb[input_label == 5] = 7  # (White: 111) Barren land
        rgb[input_label == 6] = 0  # (Black: 000) Unknown
        rgb[:, :, 0] = np.right_shift(rgb[:, :, 0], 2) % 2 * 255
        rgb[:, :, 1] = np.right_shift(rgb[:, :, 1], 1) % 2 * 255
        rgb[:, :, 2] = rgb[:, :, 2] % 2 * 255
        return rgb
    
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
    
    def print2file(self, buf, logFile):
        outfd = open(logFile, 'a')
        outfd.write(buf + '\n')
        outfd.close()