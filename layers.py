from utils import zeros, glorot
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


class Layer(object):
    def __init__(self, name='layer', verbose=True, **kwargs):
        if not name:
            layer_name = self.__class__.__name__.lower()
            name = layer_name + '_' + str(get_layer_uid(layer_name))
        else:
            layer_name = name
            name = layer_name + '_' + str(get_layer_uid(layer_name))
        self.name = name
        self.vars = {}
        self.verbose = verbose

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs=None):
        if self.verbose and inputs is not None:
            if not isinstance(inputs, list):
                tf.summary.histogram(self.name + '/inputs', inputs)
            else:
                for i, x in enumerate(inputs):
                    tf.summary.histogram(self.name + '/inputs_%d' % i, x)
        outputs = self._call(inputs)
        if self.verbose:
            tf.summary.histogram(self.name + '/outputs', outputs)
        return outputs

    def _log_vars(self):
        if self.verbose:
            for var in self.vars:
                tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class UniformSampler(Layer):
    def __init__(self, name='uniform', verbose=False, adj_list=None):
        super(UniformSampler, self).__init__(name=name, verbose=verbose)
        self.adj_list = adj_list

    def _call(self, inputs):
        ids, n_sample = inputs
        # len(id) * max_degree
        neighbors = tf.nn.embedding_lookup(self.adj_list, ids)
        neighbors = tf.transpose(
            tf.random_shuffle(
                tf.transpose(neighbors)))
        neighbors = neighbors[:, :n_sample]
        return neighbors


class GCNAgg(Layer):
    def __init__(self, name='gcn_agg', verbose=False, input_dim=None, output_dim=None,
                 act=tf.nn.relu, weight=True, dropout=0.):
        super(GCNAgg, self).__init__(name=name, verbose=verbose)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.act = act
        self.weight = weight
        self.dropout = dropout

        with tf.variable_scope(self.name):
            if self.weight:
                self.vars['weights'] = glorot([input_dim, output_dim], name='weights')
            self.vars['bias'] = zeros([output_dim], name='bias')

        self._log_vars()

    def _call(self, inputs):
        # n_sup * k, n_sup * n_sample * k, (n_sup * n_sample)
        self_vecs, neigh_vecs, n_sample = inputs
        neigh_vecs = tf.nn.dropout(neigh_vecs, 1 - self.dropout)
        self_vecs = tf.nn.dropout(self_vecs, 1 - self.dropout)

        hidden = tf.reduce_mean(tf.concat([tf.expand_dims(self_vecs, axis=1), neigh_vecs], axis=1), axis=1)
        if self.weight:
            hidden = tf.matmul(hidden, self.vars['weights'])
        hidden += self.vars['bias']
        return self.act(hidden)


class GATAgg(Layer):
    def __init__(self, name='gat_agg', verbose=False, input_dim=None, output_dim=None,
                 act=tf.nn.relu, bias=True, weight=True, dropout=0., atn_type=1, atn_drop=False):
        super(GATAgg, self).__init__(name=name, verbose=verbose)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.act = act
        self.bias = bias
        self.weight = weight
        self.dropout = dropout
        self.atn_type = atn_type
        self.atn_drop = dropout if atn_drop else 0.

        with tf.variable_scope(self.name):
            if self.weight:
                self.vars['weights'] = glorot(shape=[input_dim, output_dim], name='weights')
            else:
                assert input_dim == output_dim

            self.vars['atn_weights_1'] = glorot([output_dim, 1], name='atn_weights_1')
            self.vars['atn_weights_2'] = glorot([output_dim, 1], name='atn_weights_2')
            self.vars['atn_bias_1'] = zeros([1], name='atn_bias_1')
            self.vars['atn_bias_2'] = zeros([1], name='atn_bias_2')

            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        self._log_vars()

    def _call(self, inputs):
        # n_sup * k, n_sup * n_sample
        self_vecs, neigh_vecs, n_sample = inputs
        neigh_vecs = tf.nn.dropout(neigh_vecs, 1 - self.dropout)
        self_vecs = tf.nn.dropout(self_vecs, 1 - self.dropout)

        if self.weight:
            self_vecs = tf.matmul(self_vecs, self.vars['weights'])
            neigh_vecs = tf.reshape(
                tf.matmul(tf.reshape(neigh_vecs, [-1, self.input_dim]),
                          self.vars['weights']),
                [-1, n_sample, self.output_dim])

        # append self_vecs to neigh_vecs
        neigh_vecs = tf.concat([tf.expand_dims(self_vecs, axis=1), neigh_vecs], axis=1)
        n_neigh = n_sample + 1

        # n_sup * 1
        f_1 = tf.matmul(self_vecs, self.vars['atn_weights_1']) + self.vars['atn_bias_1']
        # n_sup * (n_sample + 1)
        f_2 = tf.reshape(
            tf.matmul(tf.reshape(neigh_vecs, [-1, self.output_dim]),
                      self.vars['atn_weights_2']),
            [-1, n_neigh]) + self.vars['atn_bias_2']
        # n_sup * (n_sample + 1)
        logits = f_1 + f_2
        scores = tf.nn.dropout(tf.nn.tanh(logits), 1 - self.atn_drop) / FLAGS.temp
        coefs = tf.nn.softmax(scores)
        output = tf.reduce_sum(tf.expand_dims(coefs, 2) * neigh_vecs, axis=1)

        if self.bias:
            output += self.vars['bias']
        return self.act(output)
