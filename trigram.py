import nltk
from nltk.util import ngrams
from collections import defaultdict
import re

def train_trigram_model(paragraphs):
    trigram_model = defaultdict(lambda: defaultdict(lambda: 0))

    for paragraph in paragraphs:
        sentences = nltk.sent_tokenize(paragraph)

        for sentence in sentences:
            words = re.findall(r'\w+', sentence.lower())            
            trigrams = ngrams(words, 3, pad_left=True, pad_right=True)

            # Update trigram counts in the model
            for w1, w2, w3 in trigrams:
                trigram_model[(w1, w2)][w3] += 1

    return trigram_model

def trigram_probability(trigram_model, w1, w2, w3):
    total_count = float(sum(trigram_model[(w1, w2)].values()))
    if total_count == 0:
        return 0
    return trigram_model[(w1, w2)][w3] / total_count

def detect_outliers_with_trigram_model(paragraph, trigram_model, threshold):
    words = re.findall(r'\w+', paragraph.lower())
    outliers = []

    for i in range(len(words) - 2):
        trigram_prob = trigram_probability(trigram_model, words[i], words[i+1], words[i+2])
        if trigram_prob < threshold:
            outliers.append(1)
        else : outliers.append(0)

    return outliers, words