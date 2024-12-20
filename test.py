import pickle


dictionary_path = "model1/dictionary.pkl"

model_dir = "model"
model_name = ["rnn_epoch_250.pkl", "rnn_epoch_500.pkl", "rnn_epoch_750.pkl", "rnn_epoch_1000.pkl"]


with open(dictionary_path, "rb") as f:
    dictionary = pickle.load(f)

word_to_cluster = dictionary["word_to_cluster"]
cluster_to_word = dictionary["cluster_to_word"]


for model_path in model_name:
    print("Cargando modelo", model_path)
    model_path = f"{model_dir}/{model_path}"
    with open(model_path, "rb") as f:
        loaded_model = pickle.load(f)

    cad = "los molinos de viento"
    generated_seq = loaded_model.generate_sequence(cad, cluster_to_word, word_to_cluster, max_length=20, temperature=1.0)
    # juntalo en un string separado por " "
    generated_seq = " ".join(generated_seq)
    print("Secuencia generada: ", generated_seq)