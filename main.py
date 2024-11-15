from get_corpus import download_corpus, preprocess_by_batches, preprocess_corpus_v2
from word2vec import Word2vecProcessor
from wordVectorQuantization import EmbeddingQuantizer
from rnn import Sequence_RNN

def preprocess_corpus(corpus, min_sentence_length=5):
    import re
    processed_corpus = []
    for line in corpus:
        line = re.sub(r'\*\d+\*', '', line) 
        line = re.sub(r'-F\w+-', '', line)
        line = re.sub(r'[^\w\s_]', '', line)
        line = line.lower()
        words = line.split()
        if len(words) >= min_sentence_length:
            processed_corpus.append(" ".join(words))

    return processed_corpus

corpus_path = "corpus/corpus_nltk.txt"
output_model_path = "model/modelo_sg.word2vec"
n_clusters = 250
epochs = 1000
n_hidden = 10
seq_len = 10

# # download_corpus(corpus_path)
# with open(corpus_path, "r", encoding="utf-8") as f:
#     raw_corpus = f.readlines()

# corpus = preprocess_corpus(raw_corpus)


corpus_path = "corpus/eswiki-latest-pages-articles.txt"
corpus = preprocess_corpus_v2(corpus_path)



print("Word2Vec")
processor = Word2vecProcessor(vector_size=100, window=5, min_count=1, sg=1)
processor.train_word2vec(corpus)
processor.save_model(output_model_path)
print(processor.find_most_similar("natural", 10))


print("Cuantizando embeddings")
embeddings = {word: processor.get_vector(word) for word in processor.model.wv.index_to_key}
print(len(embeddings))
quantizer = EmbeddingQuantizer(num_clusters=n_clusters)
quantizer.fit(embeddings)
centroids = quantizer.get_centroids()
word_to_cluster = quantizer.get_quantized_embeddings(embeddings)
cluster_to_word = {v: k for k, v in word_to_cluster.items()}

print("Diccionario de clusters:", word_to_cluster)

input_batch, target_batch = Sequence_RNN.make_batch(corpus, word_to_cluster, seq_len)

output_dim = len(centroids)
model = Sequence_RNN(embedding_matrix=centroids, hidden_dim=n_hidden, output_dim=output_dim)
model.train_model(input_batch, target_batch, num_epochs=epochs)

cad = "el grupo estatal del área"
generated_seq = model.generate_sequence(cad, cluster_to_word, word_to_cluster, max_length=15, temperature=1.0)
print("Secuencia generada:", generated_seq)