import os
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

class Word2vecProcessor:
	def __init__(self, vector_size=100, window=5, min_count=1, sg=0):
		self.vector_size = vector_size
		self.window = window
		self.min_count = min_count
		self.sg = sg

	def train_word2vec(self, corpus_path):
		print("reading corpus")
		with open(corpus_path, "r", encoding="utf-8") as f:
			corpus = f.readlines()
		print("Training")
		corpus_tokenizado = [simple_preprocess(doc) for doc in corpus]
		self.model = Word2Vec(corpus_tokenizado, vector_size=self.vector_size, window=self.window, min_count=self.min_count, sg=self.sg)
		return self.model
	
	def save_model(self, output_path):
		print("Saving")
		self.model.save(output_path)

	def load_model(self, model_path):
		self.model = Word2Vec.load(model_path)
		return self.model

	def find_most_similar(self, word, topn=10):
		return self.model.wv.most_similar(word, topn=topn)
	
	def get_vector(self, word):
		return self.model.wv[word]
	
	def get_embeddings(self):
		return {word: self.get_vector(word) for word in self.model.wv.index_to_key}