import json, numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer as TV
from sentence_transformers import SentenceTransformer
from copy import deepcopy
from time import perf_counter
from sklearn.metrics.cluster import adjusted_rand_score, rand_score
from scipy.spatial.distance import cosine

def kmeans(X, k, max_iter=100):
    """
    Performs k-means clustering on the input data.

    Parameters
    ----------
    X : numpy array, shape (n_samples, n_features)
        The input data.
    k : int
        The number of clusters.
    max_iter : int, optional
        The maximum number of iterations.
    """
    # initializing centroids to be random vectors
    centroids_idx = np.random.randint(0, X.shape[0], k)
    centroids = X[centroids_idx,:]
    # init empty data structures for algorithm
    clustered_vectors = [None] * X.shape[0]
    clustered_dict = defaultdict(list)
    iters = 0

    for _ in range(max_iter):
        iters += 1
        # clear clusters from previous iteration
        for i in range(k):
            clustered_dict[i].clear()
        # for each vector v in X choose the best cluster
        # assign to the cluster indicator array and add the vector to the cluster bin
        for i, vec in enumerate(X):
            chosen_cluster = np.argmin([np.linalg.norm((vec - cent), axis=0, keepdims=True) for cent in centroids])
            clustered_vectors[i] = chosen_cluster
            clustered_dict[chosen_cluster].append(vec)

        old_centroids = deepcopy(centroids)
        # compute the new centroids based on clusters
        for i in range(k):
            centroids[i] = np.mean(np.array(clustered_dict[i],dtype=np.float64),axis=0)
        # break loop if clusters converge
        if np.allclose(centroids,old_centroids,rtol=1e-6,atol=1e-15):
            break

    return clustered_vectors, iters

def loadData(data_file):
    data_dic = defaultdict(list)

    with open(data_file,'r') as file:
        for i,line in enumerate(file):
            line = line.split('\t')
            # line[0] == cluster , line[1] == sentence
            data_dic[line[0]].append(line[1].replace('\n',''))
    return data_dic

def extractFeatures(data_file, encoding_type):
    data_dic = loadData(data_file)
    k = len(data_dic.keys())
    corpus = []
    # flatten sentences
    for key in data_dic:
        for v in data_dic[key]:
            corpus.append(v)
    # transform to embeddings according to config
    if encoding_type == 'TFIDF':
        X = TV().fit_transform(corpus).toarray()
    else:
        X = SentenceTransformer('all-MiniLM-L6-v2').encode(corpus)
    return X, k, data_dic

def createLabelsListTrue(dict):
    retlist = []
    for i, key  in enumerate(dict):
        for _ in dict[key]:
            retlist.append(i)
    return retlist

def kmeans_cluster_and_evaluate(data_file, encoding_type):
    print(f'starting kmeans clustering and evaluation with {data_file} and encoding {encoding_type}')

    X, k, groundTruthDict = extractFeatures(data_file, encoding_type)
    labels_true = createLabelsListTrue(groundTruthDict)

    evaluation_results = {'mean_RI_score': 0.0,
                          'mean_ARI_score': 0.0}
    for _ in range(10):
        result,iters = kmeans(X, k, 100)

        evaluation_results['mean_ARI_score'] += adjusted_rand_score(labels_true, result)
        evaluation_results['mean_RI_score'] += rand_score(labels_true, result)

    evaluation_results['mean_ARI_score'] /= 10
    evaluation_results['mean_RI_score'] /= 10

    return evaluation_results

if __name__ == '__main__':
    with open('config.json', 'r') as json_file:
        config = json.load(json_file)
    start = perf_counter()
    results = kmeans_cluster_and_evaluate(config['data'], config["encoding_type"])

    for k, v in results.items():
        print(k, v)
    end = perf_counter()

    print(f'time = {(end - start)/60.0} minutes')
