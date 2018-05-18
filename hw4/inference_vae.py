import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
from model import VAE
import os, re, glob
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import skimage
import skimage.io
import argparse

import warnings
warnings.simplefilter('ignore')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--testing_images', type=str, help='Folder name of testing images')
    parser.add_argument('--gen_from', type=str, help='Folder name of the model')
    parser.add_argument('--output_images', type=str, help='Folder name of output images')
    parser.add_argument('--bsize', default=64, type=int, help='Batch size')
    parser.add_argument('--latent_dim', default=512, type=int, help='Latent dimensions')
    parser.add_argument('--lambda_kl', default=1e-2, type=float, help='Weight for the loss of KL divergence')
    args = parser.parse_args()
    inference(args)

def inference(args):
    tf.reset_default_graph()
    with tf.Session() as sess:
        net = VAE(sess,
                  latent_dim=args.latent_dim,
                  lambda_kl=args.lambda_kl)
        net.build_model()
        net.inference(test_path=args.testing_images,
                      gen_from=args.gen_from,
                      out_path=args.output_images,
                      bsize=args.bsize)

if __name__ == '__main__':
    main()
