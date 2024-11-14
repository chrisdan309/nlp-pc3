from sklearn.cluster import KMeans
import numpy as np

class EmbeddingQuantizer:
	def __init__(self, num_clusters=5):
		self.num_clusters = num_clusters
		self.kmeans = None
		self.centroids = None
		self.labels = None

	def fit(self, embeddings):
		vectors = list(embeddings.values())
		self.kmeans = KMeans(n_clusters=self.num_clusters)
		self.labels = self.kmeans.fit_predict(vectors)
		self.centroids = self.kmeans.cluster_centers_

	def get_centroids(self):
		return self.centroids
	
	def get_quantized_embeddings(self, embeddings):
		word_to_cluster = {word: self.labels[i] for i, word in enumerate(embeddings.keys())}
		return word_to_cluster