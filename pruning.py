import os
import numpy as np

import tensorflow as tf
import argparse

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--meta_filename', type=str, default='',
                    help='meta file name of the unfrozen model')
parser.add_argument('--checkpoint', type=str, default='',
                    help='checkpoint file prefix for the unfrozen model')
parser.add_argument('--output_path', type=str, default='',
                    help='output of the pruning result')
parser.add_argument('--threshold', type=float, default='0.05',
                    help='output of the pruning result')

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
    with tf.Graph().as_default() as graph:
        saver = tf.train.import_meta_graph(args.meta_filename)
        print("%d ops in the unfrozen graph." % len(graph.as_graph_def().node))

        assign_ops = []
        with tf.Session() as sess:
            saver.restore(sess, args.checkpoint)
            variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            print("%d weights in the unfrozen graph." % len(variables))
            variable_values = sess.run(variables)
            for i, v in enumerate(variables):
                if v.name.startswith("rnnlm/") and "bias" not in v.name:
                    print(v.name)
                    value = variable_values[i]
                    b = np.abs(value) < args.threshold
                    nonzero = np.count_nonzero(b)
                    size = np.size(value)
                    print("number of almost zero elements {} in {}, %{}".format(nonzero, size, nonzero/size*100))
                    print("number of nonzero in matrix (before pruning)", np.count_nonzero(value))
                    value[b] = 0
                    print("number of nonzero in matrix (after pruning)", np.count_nonzero(value))
                    assign_op = v.assign(value)
                    assign_ops.append(assign_op)

            sess.run(assign_ops)

            saver2 = tf.train.Saver()
            saver2.save(sess, args.output_path)

if __name__ == '__main__':
    main(args)
