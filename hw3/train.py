import tensorflow as tf
import numpy as np
from BuildNetVgg16_BN import NET_VGG16_BN
import os, re
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import skimage
import skimage.io
import argparse

import warnings
warnings.simplefilter('ignore')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', default='/data/put_data/cclin/ntu/dlcv2018/hw3/train', type=str, help='Folder name of training images')
    parser.add_argument('--valid_path', default='/data/put_data/cclin/ntu/dlcv2018/hw3/validation', type=str, help='Folder name of validation images')
    parser.add_argument('--vgg_path', default='/data/put_data/cclin/ntu/dlcv2018/hw3/vgg16.npy', type=str, help='File path of the vgg16 pretrained weight')
    parser.add_argument('--dir_name', default='VGG16_FCN32s', type=str, help='The folder name for the saved models and outputs')
    parser.add_argument('--result_path', default='/data/put_data/cclin/ntu/dlcv2018/hw3/results', type=str, help='The path for the saved models and outputs')
    parser.add_argument('--fcn_stride', default=32, type=int, help='32 for FCN32s; 8 for FCN8s')
    parser.add_argument('--num_epoch', default=50, type=int, help='Number of epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    parser.add_argument('--learning_rate', default=1e-5, type=float, help='Initial learning rate')
    parser.add_argument('--patience', default=10, type=int, help='Patience for early stopping')
    args = parser.parse_args()
    train(args)

def train(args):
    tf.reset_default_graph()

    with tf.Session() as sess:
    	net = NET_VGG16_BN(sess,
    		               model_name=args.dir_name,
                           result_path=args.result_path,
    		               fcn_stride=args.fcn_stride,
    		               vgg16_npy_path=args.vgg_path)
    net.build_model()
    net.train(train_path=args.train_path,
              valid_path=args.valid_path,
              nEpochs=args.num_epoch,
              bsize=args.batch_size,
              learning_rate_start=args.learning_rate,
              patience=args.patience)

if __name__ == '__main__':
    main()
