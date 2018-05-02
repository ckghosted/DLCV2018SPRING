import tensorflow as tf
import numpy as np
from BuildNetVgg16_dilated import NET_VGG16_DILATED
import os, re
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import skimage
import skimage.io
import argparse

import warnings
warnings.simplefilter('ignore')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--testing_images', type=str, help='Folder name of testing images')
    parser.add_argument('--output_images', type=str, help='Folder name of output images')
    parser.add_argument('--gen_from', type=str, help='Folder name of the model')
    args = parser.parse_args()
    inference(args)

def inference(args):
    dir_name = 'VGG16_dilated'
    Batch_Size = 16
    tf.reset_default_graph()
    with tf.Session() as sess:
        net = NET_VGG16_DILATED(sess,
                                vgg16_npy_path=os.path.join(args.gen_from, 'vgg16.npy'),
                                model_name=dir_name)
        net.build_model()
        net.inference(test_path=args.testing_images,
                      gen_from=args.gen_from,
                      out_path=args.output_images,
                      bsize=Batch_Size)

if __name__ == '__main__':
    main()
