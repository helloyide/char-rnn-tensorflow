#!/usr/bin/env python

from __future__ import print_function

import argparse
import os
import time

from six.moves import cPickle

import numpy as np

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_dir', type=str, default='data/tinyshakespeare',
                    help='data directory containing input.txt with test examples')
parser.add_argument('--save_dir', type=str, default='save',
                    help='model directory to store checkpointed models')

args = parser.parse_args()

import tensorflow as tf
from model import Model
from utils import TextLoader


def evaluate(args):
    # TODO: train and test data could have different vocabulary, here just use the train vocab for testing
    with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)

    data_loader = TextLoader(args.data_dir, saved_args.batch_size, saved_args.seq_length)

    model = Model(saved_args, training=True)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

            start = time.time()
            state = sess.run(model.initial_state)
            all_sum_mean_loss = 0
            all_count = 0
            for b in range(data_loader.num_batches):
                # both x and y have shape (batch_size, seq_length)
                x, y = data_loader.next_batch()
                feed = {model.input_data: x, model.targets: y}

                # for LSTM each initial_state element is a tuple, for GRU it is a tensor
                if saved_args.model == "lstm":
                    for i, (c, h) in enumerate(model.initial_state):
                        feed[c] = state[i].c
                        feed[h] = state[i].h
                elif saved_args.model == "gru":
                    for i, c in enumerate(model.initial_state):
                        feed[c] = state[i]

                sum_mean_loss, count = sess.run([model.pp_sum_mean_loss, model.pp_count], feed)
                all_sum_mean_loss += sum_mean_loss
                all_count += count

            print("total perplexity", np.exp(all_sum_mean_loss / all_count))
            end = time.time()
            print("inference time (in seconds):", end - start)


if __name__ == '__main__':
    evaluate(args)
