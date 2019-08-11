from __future__ import print_function
from __future__ import division

import os
# import sys
from time import time
import json
import pickle as pkl
import numpy as np
import tensorflow as tf
from tqdm import tqdm
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# sys.path.append('.')
from utils import get_support, support_to_adj_list, choice
from print_hook import PrintHook
from models import SampleAndAggregate


flags = tf.app.flags
FLAGS = flags.FLAGS
# flags.DEFINE_string('log_dir', '', '')
flags.DEFINE_bool('kg', True, 'use kg or not')
flags.DEFINE_string('dataset', 'bc', 'bc: book-crossing, ml-1m: movielens-1m, ab: amazon-book, ml-20m: movielens-20m')
flags.DEFINE_string('model', 'ni:1', 'ni:1, ni:2')
flags.DEFINE_integer('n_repeat', 5, '')
flags.DEFINE_integer('n_epoch', 100, '')
flags.DEFINE_integer('n_eval', 40, '')
flags.DEFINE_integer('early_stopping', 10, '')

flags.DEFINE_integer('max_degree', 1024, '')
flags.DEFINE_string('hop_n_sample', '1,4', 'central_node, 1-hop neighbors, 2-hop neighbors ...')

flags.DEFINE_integer('batch_size', 1024, '')
flags.DEFINE_integer('hidden', 128, '')
flags.DEFINE_string('opt', 'adam:1e-8', 'adam:eps, adagrad:init_acc, sgd')
flags.DEFINE_float('learning_rate', 1e-3, '')
flags.DEFINE_float('dropout', 0.5, '')
flags.DEFINE_float('l2_reg', 1e-5, '')
flags.DEFINE_float('temp', 10., '')
flags.DEFINE_string('act', 'relu', '')
flags.DEFINE_float('threshold', 0.5, 'this will be replaced by the average ctr later')


class Task:
    def reset(self):
        seed = 1234
        np.random.seed(seed)
        tf.set_random_seed(seed)

        if FLAGS.kg:
            self.hop_n_sample = [int(x) for x in FLAGS.hop_n_sample.split(',')]

        self.log_dir = 'log/{}/{}/'.format(FLAGS.dataset, '08-11')

        if not os.path.isdir(self.log_dir):
            os.makedirs(self.log_dir)

        self.log_file = open(self.log_dir + '/log', 'a')

        def my_hook_out(text):
            self.log_file.write(text)
            self.log_file.flush()
            return 1, 0, text

        ph_out = PrintHook()
        ph_out.Start(my_hook_out)

        self.config = {}
        for k, v in getattr(FLAGS, '__flags').items():
            self.config[k] = getattr(FLAGS, k)
        self.config_json = json.dumps(self.config, indent=4, sort_keys=True, separators=(',', ':'))

    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.support = None
        self.n_support = None
        self.n_entity = None
        self.entity_map = None
        self.adj_list = None
        self.hop_n_sample = None

        self.epoch = None
        self.global_step = None

        self.log_dir = None
        self.log_file = None
        self.config = None
        self.config_json = None

        self.reset()

        print('#' * 80)
        print(self.config_json)
        print('#' * 80)

    def load_data(self):
        print('loading data')
        dataset = pkl.load(open('data/{}.pkl'.format(FLAGS.dataset), 'rb'))
        self.train_data = dataset['train_data']
        self.test_data = dataset['test_data']
        FLAGS.threshold = self.train_data[:, 2].mean()
        print('setting threshold =', FLAGS.threshold)

        adj_mats = dataset['adj']

        if FLAGS.kg:
            support, self.entity_map = get_support(adj_mats, self.train_data, self.test_data)
        else:
            support, self.entity_map = get_support(None, self.train_data, self.test_data)

        self.n_entity = len(self.entity_map)
        adj_list, _ = support_to_adj_list(self.n_entity, support)

        self.adj_list = []
        for al in adj_list:
            if len(al) >= FLAGS.max_degree:
                self.adj_list.append(choice(al, FLAGS.max_degree))
            else:
                if len(al):
                    self.adj_list.append(choice(al, FLAGS.max_degree, replace=True))
                else:
                    self.adj_list.append([(0, 0)] * FLAGS.max_degree)
        self.adj_list = np.array(self.adj_list, dtype=np.int32)

    def get_user_record(self):
        train_record = {}
        for u, v, _ in self.train_data:
            if u not in train_record:
                train_record[u] = set()
            train_record[u].add(v)
        test_record = {}
        for u, v, s in self.test_data:
            if s == 1:
                if u not in test_record:
                    test_record[u] = set()
                test_record[u].add(v)
        return train_record, test_record

    def train(self):
        tf.reset_default_graph()
        gpu_config = tf.ConfigProto(allow_soft_placement=True,
                                    log_device_placement=False,
                                    gpu_options={'allow_growth': True})

        with tf.Session(config=gpu_config) as sess:
            if FLAGS.n_repeat > 1:
                tf.set_random_seed(int(time()))

            model_param = {'log_dir': self.log_dir,
                           'verbose': True,
                           'n_entity': self.n_entity,
                           'adj_list': self.adj_list,
                           'hop_n_sample': self.hop_n_sample}
            model = SampleAndAggregate(**model_param)

            sess.run(tf.global_variables_initializer())

            test_scores = []

            batch_size = FLAGS.batch_size if FLAGS.batch_size > 0 else self.train_data.shape[0]
            n_batch = int(np.ceil(self.train_data.shape[0] / batch_size))

            # check_embed()
            learning_rate = FLAGS.learning_rate
            for self.epoch in range(FLAGS.n_epoch):
                np.random.shuffle(self.train_data)
                for i in tqdm(range(n_batch)):
                    _data = self.train_data[i * batch_size: (i + 1) * batch_size]
                    tr_ll, self.global_step = model.train(sess, _data, learning_rate)

                # train_auc, train_ll, train_acc = model.evaluate(sess, self.train_data)
                test_auc, test_ll, test_acc = model.evaluate(sess, self.test_data)

                print('Epoch: %04d test: auc=%.6f ll=%.6f acc=%.6f' %
                      (self.epoch, test_auc, test_ll, test_acc))

                test_scores = list(test_scores)
                test_scores.append([test_auc, test_ll, test_acc])
                test_scores = np.array(test_scores)

                if test_auc >= np.max(test_scores[:, 0]):
                    # model.save(sess, epoch=self.epoch)
                    scores = model.predict(sess, self.test_data, FLAGS.n_eval)
                    np.savetxt('{}_{}.txt'.format(FLAGS.dataset, FLAGS.model), scores)

                if self.epoch > FLAGS.early_stopping and \
                        test_scores[-1, 0] < np.mean(test_scores[-(FLAGS.early_stopping + 1): -1, 0]) and \
                        test_scores[-1, 1] > np.mean(test_scores[-(FLAGS.early_stopping + 1): -1, 1]):
                    print('Early stopping...')
                    break

            print('Optimization Finished!')
            # if FLAGS.n_repeat == 1:
            #     print(self.config_json)
            ind = np.argmax(test_scores[:, 0])
            params = tuple([ind + 1] + list(test_scores[ind]))
            print('best_iter %d, auc: %.6f, ll: %.6f, acc: %.6f' % params)
            return params

    def execute(self):
        rets = []
        for i in range(FLAGS.n_repeat):
            rets.append(self.train())
        if FLAGS.n_repeat > 1:
            print(self.config_json)
            rets = np.array(rets)
            return np.mean(rets, axis=0)
        else:
            return rets[0]


if __name__ == '__main__':
    seed = 1234
    np.random.seed(seed)
    tf.set_random_seed(seed)

    task = Task()
    task.load_data()
    task.execute()

