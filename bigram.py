import nltk
from nltk.util import ngrams
from collections import defaultdict
import re

def train_bigram_model(paragraphs):
    bigram_model = defaultdict(lambda: defaultdict(lambda: 0))

    for paragraph in paragraphs:
        sentences = nltk.sent_tokenize(paragraph)

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

def detect_outliers_with_bigram_model(paragraph, bigram_model, threshold):
    words = re.findall(r'\w+', paragraph.lower())
    outliers = []

    for i in range(len(words) - 1):
        bigram_prob = bigram_probability(bigram_model, words[i], words[i+1])
        if bigram_prob < threshold:
            outliers.append(1)
        else:
            outliers.append(0)

    return outliers, words