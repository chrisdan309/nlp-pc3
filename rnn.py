import torch
import torch.nn as nn

class Sequence_RNN(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, output_dim):
        super(Sequence_RNN, self).__init__()
        embedding_tensor = torch.tensor(embedding_matrix, dtype=torch.float32)
        self.embedding = nn.Embedding.from_pretrained(embedding_tensor)
        self.rnn = nn.RNN(input_size=embedding_tensor.size(1), hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, X):
        embeds = self.embedding(X)
        out,_ = self.rnn(embeds)
        out = self.fc(out[:,-1,:])
        return out
        
    @staticmethod
    def make_batch(corpus, word_to_cluster, seq_len):
        input_batch = []
        target_batch = []
        for sentence in corpus:
            tokenized = [word_to_cluster[word] for word in sentence.split() if word in word_to_cluster]
            for i in range(len(tokenized) - seq_len):
                input_batch.append(tokenized[i:i + seq_len])
                target_batch.append(tokenized[i + seq_len])
        return torch.LongTensor(input_batch), torch.LongTensor(target_batch)

    def train_model(self, input_batch, target_batch, num_epochs=500, learning_rate=0.001, print_every=100):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            output = self(input_batch)
            loss = criterion(output, target_batch)
            if (epoch + 1) % print_every == 0:
                print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
            loss.backward()
            optimizer.step()

    def generate_sequence(self, start_string, cluster_to_word, word_to_cluster, max_length=10, temperature=1.0):
        self.eval()
        tokens = start_string.lower().split()
        start_sequence = [word_to_cluster[word] for word in tokens if word in word_to_cluster]

        if not start_sequence:
            raise ValueError("El string de entrada no contiene palabras válidas en el vocabulario.")

        generated = start_sequence[:]
        input_seq = torch.LongTensor([start_sequence])

        for _ in range(max_length - len(start_sequence)):
            with torch.no_grad():
                output = self(input_seq)
                probabilities = nn.functional.softmax(output / temperature, dim=1).squeeze()
                next_cluster = torch.multinomial(probabilities, num_samples=1).item()
                generated.append(next_cluster)
                input_seq = torch.LongTensor([generated[-len(start_sequence):]])

        result = [cluster_to_word[c] for c in generated if c in cluster_to_word]

        initial_context = tokens
        return initial_context + result[len(initial_context):]
