from get_corpus import download_corpus
from word2vec import Word2vecProcessor
from wordVectorQuantization import EmbeddingQuantizer

corpus_path = "corpus/corpus_nltk.txt"
output_model_path = "model/modelo_sg.word2vec"

n_clusters = 1098
download_corpus(corpus_path)

print("Word2Vec")
processor = Word2vecProcessor(vector_size=100, window=5, min_count=1, sg=1)
processor.train_word2vec(corpus_path)
processor.save_model(output_model_path)
print(processor.find_most_similar("natural", 10))


print("Cuantizando embeddings")
embeddings = {word: processor.get_vector(word) for word in processor.model.wv.index_to_key}
print(len(embeddings))

quantizer = EmbeddingQuantizer(num_clusters=n_clusters)
quantizer.fit(embeddings)
centroids = quantizer.get_centroids()
word_to_cluster = quantizer.get_quantized_embeddings(embeddings)

print("Diccionario de clusters:", word_to_cluster)