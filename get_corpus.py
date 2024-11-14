import os
import nltk
from nltk.corpus import cess_esp

def download_corpus(output_file):
    nltk.download('cess_esp')

    corpus = cess_esp.sents()
    corpus_text = [" ".join(sentence) for sentence in corpus]

    os.makedirs("corpus", exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for sentence in corpus_text:
            f.write(sentence + "\n")

    print(f"Corpus guardado en {output_file}")
