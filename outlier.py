from trigram import train_trigram_model, trigram_probability
from bigram import train_bigram_model, bigram_probability 
import pandas as pd
import re

def detect_outliers_with_bigram_model(paragraph, bigram_model):
    words = re.findall(r'\w+', paragraph.lower())
    outliers = []
    probability = []

    for i in range(len(words) - 1):
        bigram_prob = bigram_probability(bigram_model, words[i], words[i+1])
        probability.append(bigram_prob)
        
    return words, probability

def detect_outliers_with_trigram_model(paragraph, trigram_model):
    words = re.findall(r'\w+', paragraph.lower())
    outliers = []
    probability = []

    for i in range(len(words) - 2):
        trigram_prob = trigram_probability(trigram_model, words[i], words[i+1], words[i+2])
        probability.append(trigram_prob)

    return words, probability

def outlier_ngram(text_column,df,threshold=0.01):
    res = {
        'words' : [],
        'probs' : [],
        'na' : [],
        'a' : [],
        'na_probs': [],
        'a_probs': [],
        '%' : None
    }

    paragraphs = list(df[text_column])
    trigram_model = train_trigram_model(paragraphs)
    bigram_model = train_bigram_model(paragraphs)
    total = 0
    out = 0

    for paragraph in paragraphs :

        b_words, b_prob = detect_outliers_with_bigram_model(paragraph, bigram_model)
        t_words, t_prob = detect_outliers_with_trigram_model(paragraph, trigram_model)
        
        probs = [(t+b)/2 for t,b in zip(t_prob,b_prob)]
        outliers = [1 if probs[i]<threshold else 0 for i in range(len(probs))]
        
        # calculating % of outliers
        out = sum(outliers)
        total += len(b_words)
        
        na = []
        a = []
        na_prob = []
        a_prob = []
        for i in range(len(outliers)):
            if not outliers[i] : 
                na.append(b_words[i])
                na_prob.append(probs[i])
            else : 
                a.append(b_words[i])
                a_prob.append(probs[i])

        res['na'].append(' '.join(na))
        res['a'].append(' '.join(a))
        res['na_probs'].append(na_prob)
        res['a_probs'].append(a_prob)
    res['%'] = (out*100)/total
    
    return res