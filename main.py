from get_corpus import download_corpus, preprocess_by_batches, preprocess_corpus_v2
from word2vec import Word2vecProcessor
from wordVectorQuantization import EmbeddingQuantizer
from rnn import Sequence_RNN
import pickle

corpus_path = "corpus/corpus_nltk.txt"
output_model_path = "model/modelo_sg.word2vec"
output_rnn_model_path = "model/trained_rnn_model.pkl"
dictionary_path = "model/dictionary.pkl"

n_clusters = 1500
# n_clusters = 200
epochs = 1000
# epochs = 10
n_hidden = 10
seq_len = 10

# # download_corpus(corpus_path)
# with open(corpus_path, "r", encoding="utf-8") as f:
#     raw_corpus = f.readlines()

# corpus = preprocess_corpus(raw_corpus)


corpus_path = "corpus/eswiki-latest-pages-articles.txt"
corpus = preprocess_corpus_v2(corpus_path, batch_size=5000)
print(len(corpus))


print("Word2Vec")
processor = Word2vecProcessor(vector_size=100, window=5, min_count=1, sg=1)
processor.train_word2vec(corpus)
processor.save_model(output_model_path)


print("Cuantizando embeddings")
embeddings = {word: processor.get_vector(word) for word in processor.model.wv.index_to_key}
print(len(embeddings))
quantizer = EmbeddingQuantizer(num_clusters=n_clusters)
quantizer.fit(embeddings)
centroids = quantizer.get_centroids()
word_to_cluster = quantizer.get_quantized_embeddings(embeddings)
cluster_to_word = {v: k for k, v in word_to_cluster.items()}

dictionary = {
    "word_to_cluster": word_to_cluster,
    "cluster_to_word": cluster_to_word,
    "centroids": centroids
}

with open(dictionary_path, "wb") as f:
    pickle.dump(dictionary, f)


print("Diccionario de clusters:", word_to_cluster)

input_batch, target_batch = Sequence_RNN.make_batch(corpus, word_to_cluster, seq_len)

output_dim = len(centroids)
model = Sequence_RNN(embedding_matrix=centroids, hidden_dim=n_hidden, output_dim=output_dim)
model.train_model(input_batch, target_batch, num_epochs=epochs, print_every=100, save_every=250, save_path="model/")


# Guardar el modelo entrenado
with open(output_rnn_model_path, "wb") as f:
    pickle.dump(model, f)
print(f"Modelo RNN guardado en {output_rnn_model_path}")


cad = "el curso de la universidad"
generated_seq = model.generate_sequence(cad, cluster_to_word, word_to_cluster, max_length=15, temperature=1.0)
print("Secuencia generada:", generated_seq)