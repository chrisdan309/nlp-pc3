import numpy as np
from gensim.models import Word2Vec

class QuantizedWord:
    def __init__(self, vector_size=100, window=5, min_count=1, sg=0, bitlevel=1):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.sg = sg
        self.bitlevel = bitlevel
        self.model = None

    def quantize(self, vector):
        if self.bitlevel == 1:
            return np.where(vector >= 0, 1/3, -1/3)
        elif self.bitlevel == 2:
            quantized = np.zeros_like(vector)
            quantized[vector > 0.5] = 3/4
            quantized[(vector <= 0.5) & (vector >= 0)] = 1/4
            quantized[(vector < 0) & (vector >= -0.5)] = -1/4
            quantized[vector < -0.5] = -3/4
            return quantized
        else:
            raise ValueError("Unsupported bitlevel. Use 1 or 2.")

    def straight_through_estimator(self, quantized, original):
        return original  # El gradiente se propaga como si no hubiera cuantización

    def train(self, corpus):
        tokenized_corpus = [sentence.split() for sentence in corpus]
        print("Training")
        self.model = Word2Vec(
            sentences=tokenized_corpus,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            sg=self.sg,
            compute_loss=True
        )

        for epoch in range(5):
            for word in self.model.wv.index_to_key:
                original_vector = self.model.wv[word]
                quantized_vector = self.quantize(original_vector)
                updated_vector = self.straight_through_estimator(quantized_vector, original_vector)
                self.model.wv[word] = updated_vector

    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path):
        self.model = Word2Vec.load(path)

    def get_vector(self, word):
        if word in self.model.wv:
            return self.model.wv[word]
        else:
            raise KeyError(f"La palabra '{word}' no está en el vocabulario.")

    def get_embeddings(self):
        return {word: self.model.wv[word] for word in self.model.wv.index_to_key}
