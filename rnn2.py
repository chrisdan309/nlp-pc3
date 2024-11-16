import torch
import torch.nn as nn
import os
import pickle

class Sequence_RNN(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, output_dim):
        super(Sequence_RNN, self).__init__()
        embedding_tensor = torch.tensor(embedding_matrix, dtype=torch.float32)
        self.embedding = nn.Embedding.from_pretrained(embedding_tensor)
        self.rnn = nn.RNN(input_size=embedding_tensor.size(1), hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, X):
        embeds = self.embedding(X)
        out, _ = self.rnn(embeds)
        out = self.fc(out[:, -1, :])  # Usamos solo la última salida de la RNN
        return out

    @staticmethod
    def make_batch(corpus, word_to_index, seq_len):
        input_batch = []
        target_batch = []
        for sentence in corpus:
            tokenized = [word_to_index[word] for word in sentence.split() if word in word_to_index]
            for i in range(len(tokenized) - seq_len):
                input_batch.append(tokenized[i:i + seq_len])
                target_batch.append(tokenized[i + seq_len])
        return torch.LongTensor(input_batch), torch.LongTensor(target_batch)

    def train_model(self, input_batch, target_batch, num_epochs=500, learning_rate=0.001, print_every=100, save_every=250, save_path="model/"):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            output = self(input_batch)
            loss = criterion(output, target_batch)
            if (epoch + 1) % print_every == 0:
                print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
            loss.backward()
            optimizer.step()

            if (epoch + 1) % save_every == 0:
                model_save_path = os.path.join(save_path, f"rnn_epoch_{epoch + 1}.pkl")
                with open(model_save_path, "wb") as f:
                    pickle.dump(self, f)
                print(f"Modelo guardado en {model_save_path} después de {epoch + 1} épocas.")

    def generate_sequence(self, start_string, index_to_word, word_to_index, max_length=10, temperature=1.0):
        self.eval()
        tokens = start_string.lower().split()
        start_sequence = [word_to_index[word] for word in tokens if word in word_to_index]

        if not start_sequence:
            raise ValueError("El string de entrada no contiene palabras válidas en el vocabulario.")

        generated = start_sequence[:]
        input_seq = torch.LongTensor([start_sequence])

        for _ in range(max_length - len(start_sequence)):
            with torch.no_grad():
                output = self(input_seq)
                probabilities = nn.functional.softmax(output / temperature, dim=1).squeeze()
                next_index = torch.multinomial(probabilities, num_samples=1).item()
                generated.append(next_index)
                input_seq = torch.LongTensor([generated[-len(start_sequence):]])

        result = [index_to_word[idx] for idx in generated if idx in index_to_word]

        initial_context = tokens
        return initial_context + result[len(initial_context):]

# Preparar los datos y entrenar el modelo
if __name__ == "__main__":
    # Supongamos que tienes `quantized_embeddings`, `word_to_index` e `index_to_word`
    embedding_matrix = list(quantized_embeddings.values())
    word_to_index = {word: idx for idx, word in enumerate(quantized_embeddings.keys())}
    index_to_word = {idx: word for word, idx in word_to_index.items()}

    # Configuración de hiperparámetros
    hidden_dim = 10
    output_dim = len(word_to_index)  # El tamaño del vocabulario
    seq_len = 10
    epochs = 1000
    learning_rate = 0.001

    # Crear batches
    print("Preparando datos...")
    input_batch, target_batch = Sequence_RNN.make_batch(corpus, word_to_index, seq_len)

    # Crear el modelo
    print("Creando y entrenando el modelo RNN...")
    rnn_model = Sequence_RNN(embedding_matrix, hidden_dim, output_dim)
    rnn_model.train_model(input_batch, target_batch, num_epochs=epochs, learning_rate=learning_rate)

    # Probar la generación de secuencias
    print("Generando secuencia...")
    start_string = "el grupo estatal del área"
    generated_sequence = rnn_model.generate_sequence(start_string, index_to_word, word_to_index, max_length=15)
    print("Secuencia generada:", " ".join(generated_sequence))
