import os
import time

import tensorflow as tf
import argparse

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--frozen_filename', type=str, default='',
                    help='file name of the frozen pb file')
parser.add_argument('--frozen_graph_output_path', type=str, default='',
                    help='log path for tensorboard')
parser.add_argument('--meta_filename', type=str, default='',
                    help='meta file name of the unfrozen model')
parser.add_argument('--checkpoint', type=str, default='',
                    help='checkpoint file prefix for the unfrozen model')

args = parser.parse_args()


def load_frozen_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    # import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)
    return graph


def main(args):
    if args.frozen_filename:
        graph = load_frozen_graph(args.frozen_filename)
        print("%d ops in the frozen graph." % len(graph.as_graph_def().node))
        # print(graph.as_graph_def().node)

        if args.frozen_graph_output_path:
            writer = tf.summary.FileWriter(
                os.path.join(args.frozen_graph_output_path, "frozen_graph_" + args.frozen_filename))
            writer.add_graph(graph)

    elif args.meta_filename:
        with tf.Graph().as_default() as graph:
            saver = tf.train.import_meta_graph(args.meta_filename)
            print("%d ops in the unfrozen graph." % len(graph.as_graph_def().node))

            with tf.Session() as sess:
                saver.restore(sess, args.checkpoint)
                variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                print("%d weights in the unfrozen graph." % len(variables))
                # print(variables)


if __name__ == '__main__':
    main(args)
