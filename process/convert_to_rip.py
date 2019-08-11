from process.utils import *
from tqdm import tqdm
import pickle as pkl


def process_rs_data(rating_file, mapping_file, entity_set, saving=True):
    rec_data = np.loadtxt(rating_file, delimiter=',', dtype=str)
    print('raw rs data')
    print(rec_data)
    print('shape', rec_data.shape)

    print('loading mapping from item id to freebase id...')
    item_map = {}

    with open(mapping_file, 'r') as fin:
        for line in fin:
            rs_id, kg_id = line.strip().split()
            item_map[rs_id] = kg_id

    show_dict_sorted_by_value(item_map)

    print('indexing user id and item freebase id...')
    user_map = {}
    entity_map = {}
    for user, item, _ in tqdm(rec_data):
        assert item in item_map, 'item {} cannot be mapped to freebase'.format(item)
        fb_entity = item_map[item]
        if fb_entity in entity_set:
            if user not in user_map:
                user_map[user] = len(user_map)
            if fb_entity not in entity_map:
                entity_map[fb_entity] = len(entity_map)

    print('user mapping (size = {})'.format(len(user_map)))
    show_dict_sorted_by_value(user_map, _descent=False)
    print('entity mapping (size = {})'.format(len(entity_map)))
    show_dict_sorted_by_value(entity_map, _descent=False)

    print('converting rs data into ids...')
    rec_data_mapped = []
    for user, item, score in tqdm(rec_data):
        # assert (user in user_map) and (item in item_map), 'user {} or item {} not indexed'.format(user, item)
        if (user in user_map) and (item in item_map):
            fbid = item_map[item]
            if fbid in entity_set:
                uid = user_map[user]
                vid = entity_map[fbid]
                score = int(score)
                rec_data_mapped.append([uid, vid, score])

    rec_data_mapped = np.array(rec_data_mapped, dtype=np.int32)
    print('processed rs data')
    print(rec_data_mapped)

    if saving:
        np.save(rating_file + '.ratings_final.npy', rec_data_mapped)

    return user_map, entity_map, rec_data_mapped


def get_kg_entity_set(kg_file):
    print('finding kg entities...')
    _entity_set = set()

    num_lines = count_gz_num_lines(kg_file)

    with gzip.open(kg_file, 'rb') as fin:
        pbar = tqdm(total=num_lines)
        for line in fin:
            line = line.decode('utf-8')
            head, relation, tail = line.strip().split()[:3]
            hid = extract_id(head)
            tid = extract_id(tail)
            if hid not in _entity_set:
                _entity_set.add(hid)
            if tid not in _entity_set:
                _entity_set.add(hid)
            pbar.update(1)
        pbar.close()

    print('entity set (size = {})'.format(len(_entity_set)))
    return _entity_set


def process_kg_data(kg_file, entity_map, fname=None, saving=True):
    print('processing kg data...')
    _entity_map = entity_map.copy()
    _relation_map = {}
    _kg_triples = []

    num_lines = count_gz_num_lines(kg_file)

    # assert False, 'filtering infrequent relations'

    with gzip.open(kg_file, 'rb') as fin:
        pbar = tqdm(total=num_lines)
        for line in fin:
            line = line.decode("utf-8")
            head, relation, tail = line.strip().split()[:3]
            hid = extract_id(head)
            tid = extract_id(tail)
            if hid not in _entity_map:
                _entity_map[hid] = len(_entity_map)
            if tid not in _entity_map:
                _entity_map[tid] = len(_entity_map)
            if relation not in _relation_map:
                _relation_map[relation] = len(_relation_map)

            hid_mapped = _entity_map[hid]
            tid_mapped = _entity_map[tid]
            relation_mapped = _relation_map[relation]

            _kg_triples.append([hid_mapped, relation_mapped, tid_mapped])
            pbar.update(1)
        pbar.close()

    print('entity map (size = {})'.format(len(_entity_map)))
    show_dict_sorted_by_value(_entity_map, _descent=False)

    print('relation map (size = {})'.format(len(_relation_map)))
    show_dict_sorted_by_value(_relation_map, _descent=False)

    _kg_triples = np.array(_kg_triples, dtype=np.int32)
    print('processed kg data (shape = {})'.format(_kg_triples.shape))
    print(_kg_triples)
    print('max ids', np.max(_kg_triples, axis=0))

    if saving:
        np.save(fname, _kg_triples)

    return _entity_map, _relation_map, _kg_triples


def convert_to_rip(dataset):
    if dataset == 'ml-20m':
        rating_file = ml_20m_ratings_sampled
        mapping_file = ml_20m_mapping
        kg_file = ml_20m_fb_triples + '.n4'
    elif dataset == 'ab':
        rating_file = ab_ratings_sampled
        mapping_file = ab_mapping
        kg_file = ab_fb_triples + '.n4'
    else:
        raise NotImplementedError

    entity_set = get_kg_entity_set(kg_file)

    _, entity_map, _ = process_rs_data(rating_file, mapping_file, entity_set, saving=False)

    _entity_map, _, _ = process_kg_data(kg_file, entity_map, rating_file + '.n4.kg_final.npy', saving=False)

    pkl.dump(_entity_map, open('{}_rip_map.pkl'.format(dataset), 'wb'))


if __name__ == '__main__':
    convert_to_rip('ml-20m')
