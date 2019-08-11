import numpy as np
import matplotlib.pyplot as plt
import gzip

data_dir = '/data/lisa/data/rs_kg/'

ml_20m_mapping = data_dir + 'KB4Rec/ml2fb.txt'
ml_20m_ratings = data_dir + 'ml-20m/ratings.csv'
ml_20m_ratings_sampled = ml_20m_ratings + '.sampled'
ml_20m_fb_triples = data_dir + 'freebase_mar_15/fb_film_ns.gz'
ml_20m_num_lines = 20000263
ml_20m_threshold = 20
ml_20m_num_triples = 17319142

ab_mapping = data_dir + 'KB4Rec/ab2fb.txt'
ab_ratings = data_dir + 'Amazon_book/ratings_Books.csv'
ab_ratings_sampled = ab_ratings + '.sampled'
ab_fb_triples = data_dir + 'freebase_mar_15/fb_book_ns.gz'
ab_num_lines = 22507155
ab_threshold = 5
ab_num_triples = 13627947

rip_dir = '/data/milatmp1/quyanru/RippleNet/'

ml_1m_kg_final = rip_dir + 'data/movie/kg_final.npy'
ml_1m_ratings_final = rip_dir + 'data/movie/ratings_final.npy'

bc_kg_final = rip_dir + 'data/book/kg_final.npy'
bc_ratings_final = rip_dir + 'data/book/ratings_final.npy'

ml_20m_kg_final = rip_dir + 'data/movie_20m/kg_final.npy'
ml_20m_rating_final = rip_dir + 'data/movie_20m/ratings_final.npy'

ab_kg_final = rip_dir + 'data/amazon_book/kg_final.npy'
ab_rating_final = rip_dir + 'data/amazon_book/ratings_final.npy'


def plot_log_dist(values, ylabel, xlog=True, ylog=True):
    values = np.array(list(values))
    x_data = np.unique(list(values))
    y_data = np.array([len(np.where(values == x)[0]) for x in x_data])
    plt.scatter(x_data, y_data)
    if xlog:
        plt.xscale('log')
    if ylog:
        plt.yscale('log')
    plt.xlabel(('log ' if xlog else '') + ' frequency')
    plt.ylabel(('log ' if ylog else '') + '# ' + ylabel)
    plt.grid(which="both")
    plt.show()


def show_dict_sorted_by_value(_dict, _descent=True):
    _dict_arr = [[k, v] for k, v in _dict.items()]
    if _descent:
        _dict_arr = sorted(_dict_arr, key=lambda x: -1 * x[1])
    else:
        _dict_arr = sorted(_dict_arr, key=lambda x: x[1])
    if len(_dict_arr) <= 20:
        for k, v in _dict_arr:
            print(v, '\t', str(k))
    else:
        for i in range(5):
            print(_dict_arr[i][1], '\t', str(_dict_arr[i][0]))
        print('...')
        for i in range(-5, 0):
            print(_dict_arr[i][1], '\t', str(_dict_arr[i][0]))
    return _dict_arr


def extract_id(token):
    if token.startswith('<http://rdf.freebase.com/ns/'):
        return token[len('<http://rdf.freebase.com/ns/'):-1]
    else:
        return ''


def count_gz_num_lines(gz_file):
    with gzip.open(gz_file, 'rb') as fin:
        cnt = 0
        for _ in fin:
            cnt += 1
        return cnt
