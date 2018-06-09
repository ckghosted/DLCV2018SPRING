import tensorflow as tf
import numpy as np
from model import EXTRACTOR, LSTM
import os, re, glob
import skimage
import skimage.io

import matplotlib.pyplot as plt

import warnings
warnings.simplefilter('ignore')

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, help='Path of the trimmed validation videos folder')
    parser.add_argument('--gen_from', type=str, help='Folder name of the VGG16 model and the DNN model')
    parser.add_argument('--label_file', type=str, help='Path of ground-truth csv file')
    parser.add_argument('--output_folder', type=str, help='Path of the output labels folder')
    parser.add_argument('--bsize', default=32, type=int, help='Batch size')
    args = parser.parse_args()
    cnn_features = extract(args)
    inference(args, cnn_features)

def extract(args):
    tf.reset_default_graph()
    with tf.Session() as sess:
        net = EXTRACTOR(sess)
        net.build_vgg16(vgg16_npy_path=os.path.join(args.gen_from, 'vgg16.npy'))
        cnn_features = net.extract(video_path=args.video_path,
                                   label_file=args.label_file)
    return cnn_features
    
def inference(args, cnn_features):
    tf.reset_default_graph()
    with tf.Session() as sess:
        net = LSTM(sess,
                   max_seq_len=25,
                   h_dim=1024)
        net.build_model()
        net.inference(cnn_features=cnn_features,
                      label_file=args.label_file,
                      gen_from=args.gen_from,
                      out_path=args.output_folder,
                      bsize=args.bsize)

if __name__ == '__main__':
    main()