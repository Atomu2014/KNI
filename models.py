from layers import GCNAgg, UniformSampler
from utils import glorot, get_optimizer, get_act_func, zeros
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


def parse_params():
    int_param = FLAGS.model.split(':')
    int_type = 1
    if len(int_param) > 1:
        int_type = int(int_param[1])
    return int_type


class SampleAndAggregate(object):
    def __init__(self, name='ni', log_dir=None, verbose=True, n_entity=None, adj_list=None, hop_n_sample=None):
        self.name = name or self.__class__.__name__.lower()
        self.log_dir = log_dir
        self.verbose = verbose
        self.n_entity = n_entity
        self.hop_n_sample = hop_n_sample

        self.adj_list = tf.Variable(adj_list[:, :, 0], trainable=False, name='adj_list')
        self.sampler = UniformSampler(adj_list=self.adj_list)

        self.agg = GCNAgg
        self.int_type = parse_params()

        self.users = tf.placeholder(tf.int32, shape=[None], name='users')
        self.items = tf.placeholder(tf.int32, shape=[None], name='items')
        self.labels = tf.placeholder(tf.float32, shape=[None], name='labels')
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')
        self.is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
        self.dropout = tf.where(self.is_training, FLAGS.dropout, 0.)
        self.batch_size = tf.placeholder(tf.int32, shape=[], name='batch_size')

        # [1, hop1, hop1*hop2, ...]
        self.support_sizes = None
        self.user_neighbors = None
        self.item_neighbors = None

        self.embed = None
        self.outputs = None
        self.scores = None
        self.optimizer = None
        self.opt_param = FLAGS.opt.split(':')
        self.loss = None
        self.vars = []
        self.global_step = None
        self.train_op = None
        self.summary_op = None

        self.build()

    def build(self):
        zero_embed = tf.Variable(tf.zeros([1, FLAGS.hidden]), dtype=tf.float32, trainable=False, name='dummy_node')
        embed = glorot([self.n_entity, FLAGS.hidden], name='embed')
        self.embed = tf.concat((zero_embed, embed), axis=0)

        support_size = 1
        self.user_neighbors = [self.users]
        self.item_neighbors = [self.items]
        self.support_sizes = [support_size]
        for i in range(1, len(self.hop_n_sample)):
            n_sample = self.hop_n_sample[i]
            user_hop_i = self.sampler((self.user_neighbors[-1], n_sample))
            item_hop_i = self.sampler((self.item_neighbors[-1], n_sample))
            support_size *= n_sample
            self.user_neighbors.append(tf.reshape(user_hop_i, [self.batch_size * support_size]))
            self.item_neighbors.append(tf.reshape(item_hop_i, [self.batch_size * support_size]))
            self.support_sizes.append(support_size)

        user_hidden = [tf.nn.embedding_lookup(self.embed, hop_i) for hop_i in self.user_neighbors]
        item_hidden = [tf.nn.embedding_lookup(self.embed, hop_i) for hop_i in self.item_neighbors]

        for n_hop in range(len(self.hop_n_sample) - 2, -1, -1):
            agg_param = {
                'input_dim': FLAGS.hidden,
                'output_dim': FLAGS.hidden,
                'act': get_act_func() if n_hop else lambda x: x,
                'weight': n_hop != (len(self.hop_n_sample) - 2),
                'dropout': self.dropout,
            }
            agg = self.agg(**agg_param)

            next_user_hidden = []
            next_item_hidden = []
            last_support_size = 1
            for hop in range(n_hop + 1):
                _shape = [self.batch_size * last_support_size, self.hop_n_sample[hop + 1], FLAGS.hidden]
                user_neigh_hidden = tf.reshape(user_hidden[hop + 1], _shape)
                item_neigh_hidden = tf.reshape(item_hidden[hop + 1], _shape)

                user_h = agg((user_hidden[hop], user_neigh_hidden, self.hop_n_sample[hop + 1]))
                item_h = agg((item_hidden[hop], item_neigh_hidden, self.hop_n_sample[hop + 1]))
                last_support_size *= self.hop_n_sample[hop + 1]
                next_user_hidden.append(user_h)
                next_item_hidden.append(item_h)

            if n_hop == 0:
                neighbor_size = self.hop_n_sample[1] + 1
                hidden_size = FLAGS.hidden
                Nu = tf.concat([tf.expand_dims(user_hidden[0], 1), user_neigh_hidden], axis=1)
                Nv = tf.concat([tf.expand_dims(item_hidden[0], 1), item_neigh_hidden], axis=1)

            user_hidden = next_user_hidden
            item_hidden = next_item_hidden

        Nu = tf.nn.dropout(Nu, 1 - self.dropout)
        Nv = tf.nn.dropout(Nv, 1 - self.dropout)
        logits = tf.reduce_sum(tf.expand_dims(Nu, 2) * tf.expand_dims(Nv, 1), axis=3)
        logits = tf.reshape(logits, [-1, neighbor_size * neighbor_size])
        if self.int_type == 1:
            coefs = tf.nn.softmax(logits / FLAGS.temp)
        elif self.int_type == 2:
            with tf.variable_scope('ni'):
                w1 = glorot([hidden_size, 1], name='atn_weights_1')
                w2 = glorot([hidden_size, 1], name='atn_weights_2')
                b1 = zeros([1], name='atn_bias_1')
                b2 = zeros([1], name='atn_bias_2')
            f1 = tf.reshape(tf.matmul(tf.reshape(Nu, [-1, hidden_size]), w1) + b1, [-1, neighbor_size, 1])
            f2 = tf.reshape(tf.matmul(tf.reshape(Nv, [-1, hidden_size]), w2) + b2, [-1, 1, neighbor_size])
            coefs = tf.nn.softmax(tf.nn.tanh(tf.reshape(f1 + f2, [-1, neighbor_size * neighbor_size])) / FLAGS.temp)

        self.outputs = tf.reduce_sum(logits * coefs, axis=1)
        self.scores = tf.nn.sigmoid(self.outputs)
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.outputs, labels=self.labels))

        self.vars = {var.name: var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)}

        if FLAGS.l2_reg > 0:
            for k, v in self.vars.items():
                if ('embed' in k) or ('weight' in k):
                    self.loss += FLAGS.l2_reg * tf.nn.l2_loss(v)

        self.optimizer = get_optimizer(self.opt_param, self.learning_rate)
        self.global_step = tf.train.get_or_create_global_step()
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
        self.summary_op = tf.summary.merge_all()

    def train(self, sess, data, learning_rate):
        feed_dict = {
            self.is_training: True,
            self.users: data[:, 0],
            self.items: data[:, 1],
            self.labels: data[:, 2],
            self.batch_size: data.shape[0],
            self.learning_rate: learning_rate,
        }
        _, loss, step = sess.run([self.train_op, self.loss, self.global_step], feed_dict=feed_dict)
        return loss, step

    def evaluate(self, sess, data, n_eval=None):
        n_eval = n_eval or FLAGS.n_eval
        labels = data[:, 2]
        scores = self.predict(sess, data[:, :2], n_eval)
        auc = roc_auc_score(y_true=labels, y_score=scores)
        ll = log_loss(y_true=np.float64(labels), y_pred=np.float64(scores))
        preds = [1 if i >= FLAGS.threshold else 0 for i in scores]
        acc = accuracy_score(labels, preds)
        return auc, ll, acc

    def predict(self, sess, data, n_eval):
        evaluations = []
        for _ in range(n_eval):
            scores = []
            for i in range(int(np.ceil(data.shape[0] / FLAGS.batch_size))):
                batch = data[i * FLAGS.batch_size: (i + 1) * FLAGS.batch_size]
                feed_dict = {self.is_training: False,
                             self.users: batch[:, 0],
                             self.items: batch[:, 1],
                             self.batch_size: batch.shape[0], }
                scores.extend(sess.run(self.scores, feed_dict=feed_dict))
            evaluations.append(scores)
        evaluations = np.vstack(evaluations).transpose()
        scores = evaluations.mean(axis=1)
        return scores

    def save(self, sess, epoch):
        assert sess, 'session not provided'
        saver = tf.train.Saver(self.vars, max_to_keep=5)
        save_path = saver.save(sess, self.log_dir + 'model.ckpt', global_step=epoch)
        print('model saved at', save_path)

    def load(self, sess, epoch):
        assert sess, 'session not provided'
        saver = tf.train.Saver(self.vars)
        saver.restore(sess, self.log_dir + 'model.ckpt-{}'.format(epoch))
        print('model restored from', self.log_dir)
