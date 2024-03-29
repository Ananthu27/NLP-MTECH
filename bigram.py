import nltk
from nltk.util import ngrams
from collections import defaultdict
import re

def train_bigram_model(paragraphs):
    bigram_model = defaultdict(lambda: defaultdict(lambda: 0))

    for paragraph in paragraphs:
        sentences = nltk.sent_tokenize(str(paragraph))

        for sentence in sentences:
            words = re.findall(r'\w+', sentence.lower())            
            bigrams = ngrams(words, 2, pad_left=True, pad_right=True)

            # Update bigram counts in the model
            for w1, w2 in bigrams:
                bigram_model[w1][w2] += 1

    return bigram_model

def bigram_probability(bigram_model, w1, w2):
    total_count = float(sum(bigram_model[w1].values()))
    if total_count == 0:
        return 0
    return bigram_model[w1][w2] / total_count