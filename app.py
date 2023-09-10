from flask import Flask, request, jsonify
import json
from navec import Navec
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage
from autocorrect import Speller
from scipy.spatial import distance

from utils import *

path = 'navec_hudlit_v1_12B_500K_300d_100q.tar'
navec = Navec.load(path)
speller = Speller('ru')  
banwords = load_banwords()


app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

def clusterize(data):
    preprocessed_data, data = preprocess(data, banwords, speller)

    vectors = np.array([vectorize(sentence, navec) for sentence in preprocessed_data], dtype=float)
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=4.67, linkage="complete", metric='euclidean', compute_full_tree=True)
    clustering.fit(vectors)

    linkage_matrix = np.column_stack([clustering.children_, clustering.distances_])
    linkage_matrix = linkage(clustering.children_, method="complete", metric='euclidean')

    centroids = []
    for i in range(clustering.n_clusters_):
        cluster_points = vectors[clustering.labels_ == i]
        centroid = cluster_points.mean(axis=0)
        centroids.append(centroid)

    closest_vectors = []
    for i, centroid in enumerate(centroids):
        cluster_points = vectors[clustering.labels_ == i]
        closest_vector = min(cluster_points, key=lambda x: distance.euclidean(x, centroid))
        closest_vectors.append(closest_vector)


    clusters = dict()
    j = 0
    for closest_vector in closest_vectors:
        i = 0
        for vector in vectors:
            if list(vector) == list(closest_vector):
                clusters[data[i]] = { 'cluster': j }
                answers = list()
                for label_index in range(len(clustering.labels_)):
                    if clustering.labels_[label_index] == j:
                        answers.append(data[label_index])
                    label_index += 1
                clusters[data[i]]['answers'] = answers
            i += 1
        j += 1

    return clusters

@app.route('/uploadFile', methods=['POST'])
def upload_file():
    file = request.files.get("file")
    file_content = file.read().decode('utf-8')
    file_json = json.loads(file_content)
    data = list()
    for answer in file_json['answers']:
        data.append(answer['answer'])

    return jsonify(clusterize(data))
