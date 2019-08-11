import numpy as np
import tensorflow as tf
import scipy.sparse as sp

flags = tf.app.flags
FLAGS = flags.FLAGS


def uniform(shape, scale=0.01, name=None):
    """Uniform init."""
    initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def glorot(shape, name=None, scale=1.):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0/(shape[-1]+shape[-2])) * scale
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def embed(shape, name=None):
    assert len(shape) == 2
    return uniform(shape, scale=1. / np.sqrt(shape[1]), name=name)


def he(shape, name=None, scale=1.):
    return uniform(shape, scale=6. / np.sqrt(shape[-2] * scale), name=name)


def zeros(shape, name=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def get_optimizer(opt_param, learning_rate):
    if opt_param[0] == 'sgd':
        return tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    elif opt_param[0] == 'adagrad':
        return tf.train.AdagradOptimizer(learning_rate=learning_rate,
                                         initial_accumulator_value=float(opt_param[1]))
    elif opt_param[0] == 'adam':
        return tf.train.AdamOptimizer(learning_rate=learning_rate,
                                      epsilon=float(opt_param[1]))
    else:
        raise NotImplementedError


def get_act_func():
    if FLAGS.act == 'relu':
        act_func = tf.nn.relu
    elif FLAGS.act == 'tanh':
        act_func = tf.nn.tanh
    elif FLAGS.act == 'selu':
        act_func = tf.nn.selu
    elif FLAGS.act == '':
        def act_func(x):
            return x
    else:
        raise NotImplementedError
    return act_func


def mask_softmax(scores, mask, size):
    mask_sum = tf.reduce_sum(mask, axis=1, keepdims=True)
    mask = mask / mask_sum * size
    return tf.nn.softmax(scores) * mask


def find_neighbors(adj_mat, seeds, n_hop):
    visited = set()
    for i in range(n_hop):
        seed_list = list(seeds)
        adji_coo = adj_mat.tocoo()
        _ht = np.vstack((adji_coo.row, adji_coo.col)).transpose()
        # ignore direction
        _ind = np.isin(_ht, seed_list).any(axis=1)
        candidates = set(_ht[_ind].flatten())
        candidates = candidates - seeds
        visited |= seeds
        seeds = candidates
    return visited | seeds


def filter_entity(neighbors, rs_data=None, adj_mat=None):
    print('removing unreachable entities...')
    entity_map = {-1: 0}
    # entity_map = {}
    for n in sorted(neighbors):
        entity_map[n] = len(entity_map)

    if rs_data is not None:
        for data in rs_data:
            for i in range(len(data)):
                data[i][0] = entity_map[data[i][0]]
                data[i][1] = entity_map[data[i][1]]

    if adj_mat is not None:
        adj_reduce = []
        adji_coo = adj_mat.tocoo()
        _ht = np.vstack((adji_coo.row, adji_coo.col)).transpose()
        _ind = np.isin(_ht, neighbors).all(axis=1)
        _data = adji_coo.data[_ind]
        if len(_data):
            _ht = _ht[_ind]
            for j in range(len(_ht)):
                _ht[j][0] = entity_map[_ht[j][0]]
                _ht[j][1] = entity_map[_ht[j][1]]
            adj_reduce = sp.csr_matrix(
                (_data, (_ht[:, 0], _ht[:, 1])),
                shape=(len(entity_map), len(entity_map)))
    else:
        adj_reduce = None

    return entity_map, rs_data, adj_reduce


def pre_process_adj(adj_mat, train_data, test_data):
    rs_pairs = train_data[train_data[:, 2] == 1][:, :2]
    rs_pairs = np.vstack((rs_pairs, rs_pairs[:, [1, 0]]))

    seeds = set(train_data[:, :2].flatten()) | set(test_data[:, :2].flatten())
    if adj_mat is not None:
        n_hop = len(FLAGS.hop_n_sample.split(',')) - 1
        neighbors = list(find_neighbors(adj_mat, seeds, n_hop + 1))
        entity_map, _, adj_all = filter_entity(neighbors, [train_data, test_data, rs_pairs], adj_mat)
        adj_all = [adj_all]
    else:
        print('removing unreachable entities...')
        entity_map = {-1: 0}
        # entity_map = {}
        for n in sorted(list(seeds)):
            entity_map[n] = len(entity_map)
        for data in [train_data, test_data, rs_pairs]:
            for i in range(len(data)):
                data[i][0] = entity_map[data[i][0]]
                data[i][1] = entity_map[data[i][1]]
        adj_all = []

    adj_rs = sp.csr_matrix(
        (np.ones(rs_pairs.shape[0]),
         (rs_pairs[:, 0], rs_pairs[:, 1])),
        shape=(len(entity_map), len(entity_map)))
    adj_all += [adj_rs]

    return adj_all, entity_map


def get_support(adj_mat, train_data, test_data):
    adj_all, entity_map = pre_process_adj(adj_mat, train_data, test_data)

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = np.float32(mx.data)
        shape = mx.shape
        return coords, values, shape

    support = [to_tuple(sum(adj_all))]
    return support, entity_map


def choice(a, size, replace=False):
    if replace:
        ind = np.arange(len(a))
        ind = np.random.choice(ind, size, replace=True)
    else:
        assert len(a) >= size, '{} {} {}'.format(len(a), size, replace)
        ind = np.arange(len(a))
        np.random.shuffle(ind)
        ind = ind[:size]
    ret = []
    for i in ind:
        ret.append(a[i])
    return ret


def support_to_adj_list(n_entity, support):
    adj_list = [[] for _ in range(n_entity)]
    degree = []
    for i, sup in enumerate(support):
        hts, _, _ = sup
        for h, t in hts:
            adj_list[h].append((t, i))
    for i in range(n_entity):
        degree.append(len(adj_list))
    return adj_list, degree
