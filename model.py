import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq

import numpy as np


class Model():
    def __init__(self, args, training=True):
        self.args = args
        if not training:
            args.batch_size = 1

        # choose different rnn cell 
        if args.model == 'rnn':
            cell_fn = rnn.RNNCell
        elif args.model == 'gru':
            cell_fn = rnn.GRUCell
        elif args.model == 'lstm':
            cell_fn = rnn.LSTMCell
        elif args.model == 'nas':
            cell_fn = rnn.NASCell
        else:
            raise Exception("model type not supported: {}".format(args.model))

        # warp multi layered rnn cell into one cell with dropout
        cells = []
        for _ in range(args.num_layers):
            cell = cell_fn(args.rnn_size, dtype=tf.float32)
            if training and (args.output_keep_prob < 1.0 or args.input_keep_prob < 1.0):
                cell = rnn.DropoutWrapper(cell,
                                          input_keep_prob=args.input_keep_prob,
                                          output_keep_prob=args.output_keep_prob)
            cells.append(cell)
        self.cell = cell = rnn.MultiRNNCell(cells, state_is_tuple=True)

        # input/target data (int32 since input is char-level)
        self.input_data = tf.placeholder(
            tf.int32, [args.batch_size, args.seq_length], name="input_data")
        self.targets = tf.placeholder(
            tf.int32, [args.batch_size, args.seq_length], name="targets")
        self.initial_state = cell.zero_state(args.batch_size, tf.float32)

        # softmax output layer, use softmax to classify
        with tf.variable_scope('rnnlm'):
            softmax_w = tf.get_variable("softmax_w",
                                        [args.rnn_size, args.vocab_size])
            softmax_b = tf.get_variable("softmax_b", [args.vocab_size])

        # transform input to embedding
        embedding = tf.get_variable("embedding", [args.vocab_size, args.rnn_size])
        # input_data has shape (batch_size, seq_length) which represent a character, which could be represent as a one hot vector
        # embedding has shape (vocab_size, rnn_size)
        # after embedding_lookup, each character is not a one hot vector but an embedding vector with size rnn_size
        # the shape of inputs is (batch_size, seq_length, rnn_size)
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)

        # dropout beta testing: double check which one should affect next line
        if training and args.output_keep_prob and args.output_keep_prob < 1.0:
            inputs = tf.nn.dropout(inputs, args.output_keep_prob)

        # unstack the input to fits in rnn model
        # unstack inputs to a list with length=seq_length, each element is a tensor with the shape (batch_size, rnn_size)
        inputs = tf.split(inputs, args.seq_length, 1)
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        # loop function for rnn_decoder, which take the previous i-th cell's output and generate the (i+1)-th cell's input
        # we use it to connect the different layers in not training mode
        def loop(prev, _):
            prev = tf.matmul(prev, softmax_w) + softmax_b
            # find the best char with the highest score
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            # return the embedding format of the chars
            return tf.nn.embedding_lookup(embedding, prev_symbol)

        # rnn_decoder to generate the ouputs and final state. When we are not training the model, we use the loop function.
        # outputs is a list with length=seq_length, each element is a tensor with the shape (batch_size, rnn_size)
        outputs, last_state = legacy_seq2seq.rnn_decoder(inputs, self.initial_state, cell,
                                                         loop_function=loop if not training else None, scope='rnnlm')
        # output has shape (batch_size*seq_length, rnn_size)
        output = tf.reshape(tf.concat(outputs, 1), [-1, args.rnn_size])

        # output layer
        # softmax_w has shape (rnn_size, vocab_size), softmax_b has shape (vocab_size)
        # logits has shape (batch_size*seq_length, vocab_size)
        self.logits = tf.add(tf.matmul(output, softmax_w), softmax_b, name='output_logits')
        # probs has the same shape as logits (batch_size*seq_length, vocab_size)
        self.probs = tf.nn.softmax(self.logits)

        # loss is Weighted cross-entropy loss for a sequence of logits.
        # loss has shape (batch_size*seq_length,)
        loss = legacy_seq2seq.sequence_loss_by_example(
                [self.logits],
                # targets has shape (batch_size, seq_length), reshape it to (batch_size*seq_length,)
                [tf.reshape(self.targets, [-1])],
                # weights are all same to 1 for every logit
                [tf.ones([args.batch_size * args.seq_length])])
        # Taking the average
        with tf.name_scope('cost'):
            self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length

        with tf.name_scope('loss_monitor'):
            self.pp_count = tf.Variable(1.0, name='count')
            self.pp_sum_mean_loss = tf.Variable(1.0, name='sum_mean_loss')
            update_loss_monitor = tf.group(self.pp_sum_mean_loss.assign(tf.add(self.pp_sum_mean_loss, tf.reduce_mean(loss))),
                                           self.pp_count.assign(tf.add(self.pp_count, 1)))
            with tf.control_dependencies([update_loss_monitor]):
                self.ppl = tf.exp(self.pp_sum_mean_loss / self.pp_count)

        # last_state is a list of length=num_layers, each element has shape (batch_size, rnn_size)
        self.final_state = last_state
        self.lr = tf.Variable(0.0, trainable=False)
        # tvars is a list, each element is a weight tensor
        tvars = tf.trainable_variables()

        # calculate gradients
        # grads has the same shape as tvars,
        # tvars contains softmax_w, softmax_b, embedding, cell_0/lstm_cell/kernel (400, 800), cell_0/lstm_cell/bias (800,), cell_1... cell_LAST_LAYER
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), args.grad_clip)
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(self.lr)

        # apply gradient change to the all the trainable variable.
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

        # instrument tensorboard
        tf.summary.histogram('logits', self.logits)
        tf.summary.histogram('loss', loss)
        tf.summary.scalar('train_loss', self.cost)
        tf.summary.scalar('perplexity', self.ppl)

    def sample(self, sess, chars, vocab, num=200, prime='The ', sampling_type=1):
        """
        :param chars chars is a tuple of vocabulary chars, tuple length is vocab_size
        :param vocab vocab is a dict of char and its index, dict length is vocab_size
        :param num number of characters to sample
        :param sampling_type:
            0 to use argmax at each timestep,
            1 to sample at each timestep,
            2 to sample on the space char
        """
        state = sess.run(self.cell.zero_state(1, tf.float32))
        for char in prime[:-1]:
            x = np.zeros((1, 1))
            # get vocabulary index of the char
            x[0, 0] = vocab[char]
            # input_data has shape (batch_size, seq_length), here the shape is (1,1)
            # use one char from one step seq to predicate the next char
            feed = {self.input_data: x, self.initial_state: state}
            # final_state is a list of length=num_layers, each element has shape (batch_size, rnn_size)
            [state] = sess.run([self.final_state], feed)

        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            # sample doesn't mean always take the highest probability,
            # it's random with a higher chance to pick the char with the high probability
            # np.searchsorted: Find indices where elements should be inserted to maintain order.
            return(int(np.searchsorted(t, np.random.rand(1)*s)))

        ret = prime
        char = prime[-1]
        for _ in range(num):
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            # start from prev state, which is calcualted from the prime string
            feed = {self.input_data: x, self.initial_state: state}
            # probs has shape (batch_size*seq_length, vocab_size)
            [probs, state] = sess.run([self.probs, self.final_state], feed)
            p = probs[0]

            if sampling_type == 0:
                sample = np.argmax(p)
            elif sampling_type == 2:
                if char == ' ':
                    sample = weighted_pick(p)
                else:
                    sample = np.argmax(p)
            else:  # sampling_type == 1 default:
                sample = weighted_pick(p)

            pred = chars[sample]
            ret += pred
            char = pred
        return ret
