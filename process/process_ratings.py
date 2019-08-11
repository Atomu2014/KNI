from tqdm import tqdm
from process.utils import *


def ml_20m_parser(line):
    u, v, s, t = line.strip().split(',')
    s, t = float(s), int(t)
    return u, v, s, t, 1104537600 < t < 1420070400


def ab_parser(line):
    u, v, s, t = line.strip().split(',')
    return u, v, float(s), int(t), True


def count_user_item_freq(mapping_file, rating_file, num_lines, threshold=20):
    item_set = set(np.loadtxt(mapping_file, delimiter='\t', usecols=[0], dtype=str))
    user_freq = {}
    item_freq = {}
    rating_values = []

    with open(rating_file, 'r') as fin:
        if dataset == 'ml-20m':
            fin.readline()
            parser = ml_20m_parser
        else:
            parser = ab_parser
        pbar = tqdm(total=num_lines)
        for line in fin:
            u, v, s, t, flag = parser(line)
            if flag and (v in item_set):
                user_freq[u] = user_freq.get(u, 0) + 1
                item_freq[v] = item_freq.get(v, 0) + 1
                rating_values.append(s)
            pbar.update(1)
        pbar.close()

        rating_values = np.array(rating_values)
        print('{} users, {} items, {} ratings in total'.format(len(user_freq), len(item_freq), len(rating_values)))

    show_dict_sorted_by_value(user_freq)
    plot_log_dist(user_freq.values(), 'user')

    show_dict_sorted_by_value(item_freq)
    plot_log_dist(item_freq.values(), 'item')

    plt.hist(rating_values, bins=10)
    plt.show()

    print('filtering low frequency users/items...')
    item_filtered = set()
    for item in item_freq.keys():
        if item_freq[item] >= threshold:
            item_filtered.add(item)

    user_filtered = set()
    for user in user_freq.keys():
        if user_freq[user] >= threshold:
            user_filtered.add(user)

    print('{} from {} users have ratings >= {}'.format(len(user_filtered), len(user_freq), threshold))
    print('{} from {} items have ratings >= {}'.format(len(item_filtered), len(item_freq), threshold))
    print('finish')
    return user_filtered, item_filtered


def find_pos_neg_ratings(rating_file, user_set, item_set, num_lines):
    print('finding pos/neg ratings...')
    likes = {}
    dislikes = {}
    with open(rating_file, 'r') as fin:
        if dataset == 'ml-20m':
            fin.readline()
            parser = ml_20m_parser
        else:
            parser = ab_parser
        pbar = tqdm(total=num_lines)
        for line in fin:
            u, v, s, t, flag = parser(line)
            if flag and (u in user_set) and (v in item_set):
                if s >= 4:
                    if u in likes:
                        likes[u].add(v)
                    else:
                        likes[u] = {v}
                else:
                    if u in dislikes:
                        dislikes[u].add(v)
                    else:
                        dislikes[u] = {v}
            pbar.update(1)
        pbar.close()

    print('{} users have positive ratings'.format(len(likes)))
    print('{} users have negative ratings'.format(len(dislikes)))
    print('finish')
    like_cnt = np.array([len(x) for x in likes.values()])
    dislike_cnt = np.array([len(x) for x in dislikes.values()])
    print('{} likes in total'.format(np.sum(like_cnt)))
    print('{} dislikes in total'.format(np.sum(dislike_cnt)))
    plot_log_dist(like_cnt, 'like')
    plot_log_dist(dislike_cnt, 'dislike')
    return likes, dislikes


def generate_samples(rating_file, likes, dislikes, user_set, item_set):
    pos_items = set()
    for items in likes.values():
        for item in items:
            pos_items.add(item)
    print('{} from {} items have positive ratings'.format(len(pos_items), len(item_set)))

    print('generating pos/neg samples...')
    pos_samples = {}
    neg_samples = {}
    pbar = tqdm(total=len(likes))
    for user, items in likes.items():
        pos_samples[user] = items
        item_reviewed = items | dislikes.get(user, set())
        candidates = list(pos_items - item_reviewed)
        np.random.shuffle(candidates)
        neg_samples[user] = set(candidates[:len(items)])
        pbar.update(1)
    pbar.close()
    print('finish')

    print('generating interactions...')
    ratings_sampled = []
    users_sampled = set()
    items_sampled = set()

    pbar = tqdm(total=len(pos_samples))
    for user, items in pos_samples.items():
        users_sampled.add(user)
        for item in items:
            items_sampled.add(item)
            ratings_sampled.append([user, item, 1])
        pbar.update(1)
    pbar.close()

    pbar = tqdm(total=len(neg_samples))
    for user, items in neg_samples.items():
        users_sampled.add(user)
        for item in items:
            items_sampled.add(item)
            ratings_sampled.append([user, item, 0])
        pbar.update(1)
    pbar.close()

    print('{} from {} users, {} from {} items in samples, {} interactions'.
          format(len(users_sampled), len(user_set), len(items_sampled), len(item_set), len(ratings_sampled)))
    print('finish')

    np.random.shuffle(ratings_sampled)

    with open(rating_file + '.sampled', 'w') as fout:
        pbar = tqdm(total=len(ratings_sampled))
        for u, v, s in ratings_sampled:
            fout.write('{},{},{}\n'.format(u, v, s))
            pbar.update(1)
        pbar.close()

    with open(rating_file + '.users_sampled', 'w') as fout:
        for user in users_sampled:
            fout.write('{}\n'.format(user))

    with open(rating_file + '.items_sampled', 'w') as fout:
        for item in items_sampled:
            fout.write('{}\n'.format(item))


def process_ratings():
    if dataset == 'ml-20m':
        mapping_file = ml_20m_mapping
        rating_file = ml_20m_ratings
        num_lines = ml_20m_num_lines
        threshold = ml_20m_threshold
    elif dataset == 'ab':
        mapping_file = ab_mapping
        rating_file = ab_ratings
        num_lines = ab_num_lines
        threshold = ab_threshold
    else:
        raise NotImplementedError

    user_filtered, item_filtered = count_user_item_freq(mapping_file, rating_file, num_lines, threshold=threshold)

    likes, dislikes = find_pos_neg_ratings(rating_file, user_filtered, item_filtered, num_lines)

    generate_samples(rating_file, likes, dislikes, user_filtered, item_filtered)


def scan_fb_triples(fb_file, _callback):
    if dataset == 'ml-20m':
        pbar = tqdm(total=ml_20m_num_triples)
    else:
        pbar = tqdm(total=ab_num_triples)
    with gzip.open(fb_file, 'rb') as _fin:
        _cnt = 0
        for _line in _fin:
            _line = _line.decode('utf-8')
            _head, _relation, _tail = _line.strip().split()[:3]
            _hid, _tid = extract_id(_head), extract_id(_tail)
            _callback(**{
                'line': _line,
                'head': _head,
                'relation': _relation,
                'tail': _tail,
                'hid': _hid,
                'tid': _tid,
            })
            pbar.update(1)
        pbar.close()


def expand(fb_file, _visited, _seeds):
    assert (len(_visited & _seeds) == 0) and ('' not in _visited) and ('' not in _seeds)
    print('assert True')
    _candidates = set()
    _VUS = _visited.union(_seeds)

    def foo(**kwargs):
        hid = kwargs['hid']
        tid = kwargs['tid']
        if (hid in _seeds) or (tid in _seeds):
            for _id in [hid, tid]:
                if _id != '' and _id not in _VUS:
                    _candidates.add(_id)

    scan_fb_triples(fb_file, foo)
    print(len(_candidates), 'candidates are explored')

    return _candidates


def count_node_freq(fb_file, _visited, _candidates, _threshold=20):
    assert (len(_visited & _candidates) == 0) and ('' not in _visited) and ('' not in _candidates)
    print('assert True')
    _node_freq = {}
    _VUC = _visited.union(_candidates)
    print(len(_visited), 'visited,', len(_candidates), 'candidates,', len(_VUC), 'in total')

    def foo(**kwargs):
        hid = kwargs['hid']
        tid = kwargs['tid']
        if (hid in _VUC) and (tid in _VUC):
            for _id in [hid, tid]:
                if _id in _candidates:
                    _node_freq[_id] = _node_freq.get(_id, 0) + 1

    scan_fb_triples(fb_file, foo)
    print(len(_node_freq), 'nodes counted')

    show_dict_sorted_by_value(_node_freq)

    _candidates_filtered = set()
    for k, v in _node_freq.items():
        if v >= _threshold:
            _candidates_filtered.add(k)

    print('threshold =', _threshold)
    print(len(_candidates_filtered), 'candidates remaining')

    return _node_freq, _candidates_filtered


def count_link_freq(fb_file, _nodes, _threshold=5000):
    _link_freq = {}

    def foo(**kwargs):
        hid = kwargs['hid']
        tid = kwargs['tid']
        relation = kwargs['relation']

        if (hid in _nodes) and (tid in _nodes):
            _link_freq[relation] = _link_freq.get(relation, 0) + 1

    scan_fb_triples(fb_file, foo)

    print(len(_link_freq), 'links in total')

    show_dict_sorted_by_value(_link_freq)

    _links_filtered = set()
    for k, v in _link_freq.items():
        if v >= _threshold:
            _links_filtered.add(k)

    print('threshold =', _threshold)
    print(len(_links_filtered), 'links remaining')

    return _link_freq, _links_filtered


def expand_neighborhood(fb_file, _visited, _seeds, _fname, _node_threshold=20, _link_threshold=5000):
    assert (len(_visited & _seeds) == 0) and ('' not in _visited) and ('' not in _seeds)
    print('assert True')
    print(len(_visited), 'visited,', len(_seeds), 'seeds, node threshold =', _node_threshold, ', link threshold =',
          _link_threshold)

    print('\n...expand...')
    _candidates = expand(fb_file, _visited, _seeds)

    _VUS = _visited.union(_seeds)

    print('\n...filter candidates...')
    _node_freq, _candidates_filtered = count_node_freq(fb_file, _VUS, _candidates, _node_threshold)

    plot_log_dist(_node_freq.values(), 'node')

    _VUSUC = _VUS.union(_candidates_filtered)

    print('\n...filter links...')
    _link_freq, _links_filtered = count_link_freq(fb_file, _VUSUC, _link_threshold)

    plot_log_dist(_link_freq.values(), 'link', ylog=False)

    print('\n...dump triples...')
    with gzip.open(fb_file + _fname, 'wb') as _fout:
        def foo(**kwargs):
            line = kwargs['line']
            hid = kwargs['hid']
            tid = kwargs['tid']
            relation = kwargs['relation']

            if (hid in _VUSUC) and (tid in _VUSUC) and (relation in _links_filtered):
                _fout.write(line.encode('utf-8'))

        scan_fb_triples(fb_file, foo)
        print('\n')

    print(len(_candidates_filtered), 'candidates added,', len(_VUSUC), 'node in total')

    return _candidates_filtered, _links_filtered


def process_freebase():
    if dataset == 'ml-20m':
        rating_file = ml_20m_ratings
        mapping_file = ml_20m_mapping
        fb_file = ml_20m_fb_triples
        threshold = ml_20m_threshold
    elif dataset == 'ab':
        rating_file = ab_ratings
        mapping_file = ab_mapping
        fb_file = ab_fb_triples
        threshold = ab_threshold
    else:
        raise NotImplementedError

    items_sampled = set(np.loadtxt(rating_file + '.items_sampled', dtype=str))
    item_2_fb_id = {}
    with open(mapping_file, 'r') as fin:
        for line in fin:
            rs_id, kg_id = line.strip().split()
            item_2_fb_id[rs_id] = kg_id
    seeds = set()
    for item in items_sampled:
        if item in item_2_fb_id:
            seeds.add(item_2_fb_id[item])
        else:
            print('{} not in mapping'.format(item))
    print('{} seeds'.format(len(seeds)))

    n1_nodes, n1_links = expand_neighborhood(fb_file, set(), seeds, '.n1', threshold)

    n2_nodes, n2_links = expand_neighborhood(fb_file, seeds, n1_nodes, '.n2', threshold)

    SUN1 = seeds.union(n1_nodes)
    n3_nodes, n3_links = expand_neighborhood(fb_file, SUN1, n2_nodes, '.n3', threshold)

    SUN2 = SUN1.union(n2_nodes)
    n4_nodes, n4_links = expand_neighborhood(fb_file, SUN2, n3_nodes, '.n4', threshold)


if __name__ == '__main__':
    np.random.seed(1234)
    dataset = 'ab'
    process_ratings()
    process_freebase()
