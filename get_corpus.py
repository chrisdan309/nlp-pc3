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

def preprocess_by_batches(batch_size, corpus_path):
	corpus = " "
	cont = 0
	with open(corpus_path, 'r', encoding='utf-8') as f:
		for line in f:
			corpus += line.lower()
			cont += 1
			if cont == batch_size:
				break
	return corpus


def preprocess_corpus_v2(corpus_path, min_sentence_length=5, batch_size = 5000):
	import re
	cont = 0
	processed_corpus = []
	with open(corpus_path, 'r', encoding='utf-8') as f:
		for line in f:
			cont += 1
			line = re.sub(r'\*\d+\*', '', line) 
			line = re.sub(r'-F\w+-', '', line)
			line = re.sub(r'[^\w\s_]', '', line)
			line = line.lower()
			words = line.split()
			if len(words) >= min_sentence_length:
				processed_corpus.append(" ".join(words))
			if cont == batch_size:
				break

	return processed_corpus