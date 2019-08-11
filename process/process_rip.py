from process.utils import *
from tqdm import tqdm
import networkx as nx
import pickle as pkl


def split_train_test(rating_np):
    """
    same as RippleNet
    """
    print('splitting dataset ...')
    # train:eval:test = 6:2:2
    eval_ratio = 0.2
    test_ratio = 0.2
    n_ratings = rating_np.shape[0]

    eval_indices = np.random.choice(list(range(n_ratings)), size=int(n_ratings * eval_ratio), replace=False)
    left = set(range(n_ratings)) - set(eval_indices)
    test_indices = np.random.choice(list(left), size=int(n_ratings * test_ratio), replace=False)
    train_indices = list(left - set(test_indices))
    # print(len(train_indices), len(eval_indices), len(test_indices))

    # traverse training data, only keeping the users with positive ratings
    user_history_dict = dict()
    for i in train_indices:
        user = rating_np[i][0]
        item = rating_np[i][1]
        rating = rating_np[i][2]
        if rating == 1:
            if user not in user_history_dict:
                user_history_dict[user] = []
            user_history_dict[user].append(item)

    train_indices = [i for i in train_indices if rating_np[i][0] in user_history_dict]
    eval_indices = [i for i in eval_indices if rating_np[i][0] in user_history_dict]
    test_indices = [i for i in test_indices if rating_np[i][0] in user_history_dict]
    # print(len(train_indices), len(eval_indices), len(test_indices))

    train_data = rating_np[train_indices]
    eval_data = rating_np[eval_indices]
    test_data = rating_np[test_indices]

    return train_data, eval_data, test_data, user_history_dict


def process_rs_data(rating_file):
    rec_data = np.load(rating_file)
    print('raw rs data')
    print(rec_data)
    print('shape', rec_data.shape)

    entity_map = {}
    for user, item, _ in tqdm(rec_data):
        user_name = 'u{}'.format(user)
        item_name = 'v{}'.format(item)
        if user_name not in entity_map:
            entity_map[user_name] = len(entity_map)
        if item_name not in entity_map:
            entity_map[item_name] = len(entity_map)

    print('entity map (size = {})'.format(len(entity_map)))
    show_dict_sorted_by_value(entity_map)

    print('converting rs data into ids...')
    rec_data_mapped = []
    for user, item, score in tqdm(rec_data):
        user_name = 'u{}'.format(user)
        item_name = 'v{}'.format(item)
        uid = entity_map[user_name]
        vid = entity_map[item_name]
        score = int(score)
        rec_data_mapped.append([uid, vid, score])

    rec_data_mapped = np.array(rec_data_mapped, dtype=np.int32)
    print('processed rs data')
    print(rec_data_mapped)

    return entity_map, rec_data_mapped


# def find_neighbors(kg_np, seeds, n_hop=1):
#     print('finding neighbors...')
#
#     visited = set()
#     neighbors = []
#     for i in range(n_hop):
#         # candidates = set()
#         # for h, r, t in tqdm(kg_np):
#         #     if (h in seeds) or (t in seeds):
#         #         candidates.add(h)
#         #         candidates.add(t)
#         # candidates -= seeds
#         candidates = multi_proc(1, pre_proc, part_job, post_proc, kg_np=kg_np, seeds=seeds)
#         print('hop-{}, {} candidate'.format(i+1, len(candidates)))
#         visited |= seeds
#         seeds = candidates
#         neighbors.append(visited | seeds)
#     return neighbors


def process_kg_data(kg_np, entity_map, threshold=100):
    _entity_map = entity_map.copy()
    _relation_map = {}
    _kg_triples = []

    print('filtering relations appearing less than {} times'.format(threshold))
    relation_freq = {}
    for _, relation, _ in kg_np:
        relation_freq[relation] = relation_freq.get(relation, 0) + 1
    # show_dict_sorted_by_value(relation_freq)

    for head, relation, tail in tqdm(kg_np):
        if relation_freq[relation] < threshold:
            continue
        head = 'v{}'.format(head)
        tail = 'v{}'.format(tail)
        if head not in _entity_map:
            _entity_map[head] = len(_entity_map)
        if tail not in _entity_map:
            _entity_map[tail] = len(_entity_map)
        if relation not in _relation_map:
            _relation_map[relation] = len(_relation_map)

        hid = _entity_map[head]
        tid = _entity_map[tail]
        rid = _relation_map[relation]

        _kg_triples.append([hid, rid, tid])

    print('entity map, size = {}'.format(len(_entity_map)))
    show_dict_sorted_by_value(_entity_map, _descent=False)

    print('relation map, size = {}'.format(len(_relation_map)))
    show_dict_sorted_by_value(_relation_map, _descent=False)

    _kg_triples = np.array(_kg_triples, dtype=np.int32)
    print('processed kg data, shape = {}'.format(_kg_triples.shape))
    print(_kg_triples)
    print('max ids', np.max(_kg_triples, axis=0))

    return _entity_map, _relation_map, _kg_triples


def inverse_kg(_kg, mode=0):
    _kg_inv = {}
    for h in _kg.keys():
        for t in _kg[h]:
            if mode == 0:
                if t in _kg_inv:
                    _kg_inv[t].add(h)
                else:
                    _kg_inv[t] = {h}
            elif mode == 1:
                t, r = t
                if t in _kg_inv:
                    _kg_inv[t].add((h, r))
                else:
                    _kg_inv[t] = {(h, r)}
            else:
                raise NotImplementedError
    for k in _kg_inv.keys():
        _kg_inv[k] = sorted(list(_kg_inv[k]))
        # if mode == 1:
        #     _kg_inv[k] = list(map(lambda x: [x[0], x[1]], _kg_inv[k]))
    return _kg_inv


def complete_kg(_kg, _entity_map):
    _kg_complete = {}
    for node in range(len(_entity_map)):
        if node in _kg:
            _kg_complete[node] = _kg[node]
        else:
            _kg_complete[node] = []
    return _kg_complete


def kg_triples_to_adj(_entity_map, _relation_map, _kg_triples, ):
    print('constructing kg dicts...')
    _kg_simple = {}
    _kg_relational = [{} for _ in range(len(_relation_map))]
    for hid, rid, tid in tqdm(_kg_triples):
        if hid in _kg_simple:
            _kg_simple[hid].add(tid)
        else:
            _kg_simple[hid] = {tid}
        kgi = _kg_relational[rid]
        if hid in kgi:
            kgi[hid].add(tid)
        else:
            kgi[hid] = {tid}

    for k in _kg_simple.keys():
        _kg_simple[k] = list(_kg_simple[k])

    for kgi in _kg_relational:
        for k in kgi.keys():
            kgi[k] = list(kgi[k])

    kgs = {'kg_simple': _kg_simple,
           'kg_relational': _kg_relational, }

    for k in ['directed', 'undirected', 'bi-directed']:
        if k == 'undirected':
            create_using = nx.Graph()
        else:
            create_using = nx.DiGraph()

        print('constructing {} adjacent matrices...'.format(k))

        _adj_simple = nx.adj_matrix(
            nx.from_dict_of_lists(
                d=complete_kg(_kg_simple, _entity_map),
                create_using=create_using)
        ).astype(np.int32)

        print('all relations: {} edges, shape = {}'.format(len(_adj_simple.data), _adj_simple.shape))
        if k == 'bi-directed':
            _adj_simple_inv = nx.adj_matrix(
                nx.from_dict_of_lists(
                    d=complete_kg(inverse_kg(_kg_simple), _entity_map),
                    create_using=create_using)
            ).astype(np.int32)
            _adj_simple = [_adj_simple, _adj_simple_inv]
            print('all inverse relations: {} edges, shape = {}'.format(len(_adj_simple_inv.data), _adj_simple_inv.shape))

        _adj_relational = []
        for kgi in tqdm(_kg_relational):
            _adj_relational.append(nx.adj_matrix(
                nx.from_dict_of_lists(
                    d=complete_kg(kgi, _entity_map),
                    create_using=create_using)
            ).astype(np.int32))

            if k == 'bi-directed':
                _adj_relational.append(nx.adj_matrix(
                    nx.from_dict_of_lists(
                        d=complete_kg(inverse_kg(kgi), _entity_map),
                        create_using=create_using)
                ).astype(np.int32))

        for i, adji in enumerate(_adj_relational):
            print('relation {}: {} edges, shape = {}'.format(i, len(adji.data), adji.shape))

        kgs[k] = {
            'adj_simple': _adj_simple,
            'adj_relational': _adj_relational
        }

    return kgs


def convert(_dataset):
    if _dataset == 'ml-1m':
        rating_file = ml_1m_ratings_final
        kg_file = ml_1m_kg_final
    elif _dataset == 'bc':
        rating_file = bc_ratings_final
        kg_file = bc_kg_final
    elif _dataset == 'ml-20m':
        rating_file = ml_20m_rating_final
        kg_file = ml_20m_kg_final
    elif _dataset == 'ab':
        rating_file = ab_rating_final
        kg_file = ab_kg_final
    else:
        raise NotImplementedError

    entity_map, rec_data = process_rs_data(rating_file)

    train_data, eval_data, test_data, user_history_dict = split_train_test(rec_data)

    kg_np = np.load(kg_file)
    print('raw kg triples, shape = {}'.format(kg_np.shape))

    _entity_map, _relation_map, _kg_triples = process_kg_data(kg_np, entity_map)

    _kgs = kg_triples_to_adj(_entity_map, _relation_map, _kg_triples)

    dataset = {
        'train_data': train_data,
        'eval_data': eval_data,
        'test_data': test_data,
        'user_history_dict': user_history_dict,
        'entity_map': _entity_map,
        'relation_map': _relation_map,
        'kg_triples': _kg_triples,
    }

    for k, v in _kgs.items():
        dataset[k] = v

    file_name = '../data/{}.pkl'.format(_dataset)
    print('pickling to {}...'.format(file_name))
    print(train_data.dtype, eval_data.dtype, test_data.dtype)
    for k, v in user_history_dict.items():
        print(type(k), type(v[0]))
        break
    for k, v in _entity_map.items():
        print(type(k), type(v))
        break
    for k, v in _relation_map.items():
        print(type(k), type(v))
        break
    for k, v in _kgs['kg_simple'].items():
        print(type(k), type(v[0]))
        break
    for kgi in _kgs['kg_relational']:
        for k, v in kgi.items():
            print(type(k), type(v[0]))
            break
        break
    print(_kgs['directed']['adj_simple'].dtype)
    for adji in _kgs['directed']['adj_relational']:
        print(adji.dtype)
        break

    pkl.dump(dataset, open(file_name, 'wb'))


if __name__ == '__main__':
    np.random.seed(555)
    convert('ab')
