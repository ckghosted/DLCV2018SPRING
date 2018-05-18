import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
from model_gan import GAN, INFOGAN
import os, re, glob
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import skimage
import skimage.io
import argparse

import warnings
warnings.simplefilter('ignore')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_from', type=str, help='Folder name of the model')
    parser.add_argument('--output_images', type=str, help='Folder name of output images')
    parser.add_argument('--bsize', default=64, type=int, help='Batch size')
    parser.add_argument('--random_dim', default=64, type=int, help='Input dimension of the random noise')
    parser.add_argument('--n_cat', default=5, type=int, help='Number of categorical latent codes')
    parser.add_argument('--cat_dim', default=10, type=int, help='Dimension of each categorical latent code')
    parser.add_argument('--cont_dim', default=0, type=int, help='Number of continuous latent codes')
    args = parser.parse_args()
    inference(args)

def inference(args):
    tf.reset_default_graph()
    with tf.Session() as sess:
        net = INFOGAN(sess,
                      random_dim=args.random_dim,
                      n_cat=args.n_cat,
                      cat_dim=args.cat_dim,
                      cont_dim=args.cont_dim)
        net.build_model()
        net.inference(gen_from=args.gen_from,
                      out_path=args.output_images,
                      bsize=args.bsize)

if __name__ == '__main__':
    main()
