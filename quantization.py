
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--saved_model_dir', type=str, default='',
                    help='The directory should contains a file with the name saved_model.pbtxt or saved_model.pb')
parser.add_argument('--frozen_model_file', type=str, default='',
                    help='The pb file name of the frozen model')
parser.add_argument('--output_file', type=str, default='',
                    help='where to store the converted model')
parser.add_argument('--input_node_name', type=str, default='',
                    help='the input node name in graph')
parser.add_argument('--output_node_name', type=str, default='',
                    help='the output node name in graph')

args = parser.parse_args()

# get error about "toco_from_protos: not found"?
# https://stackoverflow.com/a/53075836/1943272
def main(args):
    # if get tags could not be found in SavedModel error with the from_saved_model() call
    # make sure the .pb file is not from the frozen model, which is different from the .pb file of a SavedModel,
    # be sure your .pb file was generated in the correct way as it is described in the docs:
    # https://www.tensorflow.org/guide/saved_model.
    if args.saved_model_dir:
        converter = tf.contrib.lite.TFLiteConverter.from_saved_model(args.saved_model_dir)
    else:
        converter = tf.contrib.lite.TFLiteConverter.from_frozen_graph(args.frozen_model_file, [args.input_node_name], [args.output_node_name])

    converter.post_training_quantize = True
    tflite_quantized_model = converter.convert()
    open(args.output_file, "wb").write(tflite_quantized_model)


if __name__ == '__main__':
    main(args)
