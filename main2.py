from get_corpus import preprocess_by_batches, preprocess_corpus_v2
from rnn2 import Sequence_RNN
from word2vecQ import QuantizedWord
import pickle
import re
import numpy as np
def preprocess_corpus(corpus, min_sentence_length=5):
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

corpus_path = "corpus/eswiki-latest-pages-articles.txt"
output_model_path = "model/quantized_model.word2vec"
output_rnn_model_path = "model/trained_rnn_model.pkl"

# Hiperparámetros
epochs = 1000
n_hidden = 10
seq_len = 10
vector_size = 50
bitlevel = 2  # Cuantización de 2 bits

# Preprocesar el corpus
print("Preprocesando el corpus...")
corpus = preprocess_corpus_v2(corpus_path, batch_size=1000)

# Entrenar el modelo QuantizedWord
print("Entrenando modelo Word2Vec cuantizado...")
quantized_processor = QuantizedWord(
    vector_size=vector_size,
    window=5,
    min_count=1,
    sg=1,
    bitlevel=bitlevel
)
quantized_processor.train(corpus)

# Guardar el modelo entrenado
quantized_processor.save_model(output_model_path)

# Obtener los vectores cuantizados
print("Obteniendo embeddings cuantizados...")
quantized_embeddings = {
    word: quantized_processor.get_vector(word)
    for word in quantized_processor.model.wv.index_to_key
}
print(f"Total de palabras en el vocabulario: {len(quantized_embeddings)}")

# Crear el diccionario de vectores (directo del modelo cuantizado)
word_to_index = {word: idx for idx, word in enumerate(quantized_embeddings.keys())}
index_to_word = {idx: word for word, idx in word_to_index.items()}

# Guardar el diccionario
dictionary = {
    "word_to_index": word_to_index,
    "index_to_word": index_to_word
}

dictionary_path = "model/dictionary.pkl"
with open(dictionary_path, "wb") as f:
    pickle.dump(dictionary, f)

print("Diccionario guardado en", dictionary_path)

# Crear batches para entrenar la RNN
print("Preparando datos para la RNN...")
input_batch, target_batch = Sequence_RNN.make_batch(corpus, word_to_index, seq_len)

# Configurar y entrenar la RNN
embedding_matrix = np.array(list(quantized_embeddings.values()), dtype=np.float32)

output_dim = len(word_to_index)

model = Sequence_RNN(embedding_matrix=embedding_matrix, hidden_dim=n_hidden, output_dim=output_dim)

print("Entrenando modelo RNN...")
model.train_model(input_batch, target_batch, num_epochs=epochs, print_every=100, save_every=250, save_path="model/")

# Guardar el modelo RNN entrenado
with open(output_rnn_model_path, "wb") as f:
    pickle.dump(model, f)
print(f"Modelo RNN guardado en {output_rnn_model_path}")

# Probar la generación de secuencias con el modelo RNN
print("Probando la generación de secuencias...")
cad = "el grupo estatal del área"
generated_seq = model.generate_sequence(cad, index_to_word, word_to_index, max_length=15, temperature=1.0)
print("Secuencia generada:", " ".join(generated_seq))
