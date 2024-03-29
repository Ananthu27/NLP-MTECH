from trigram import train_trigram_model, trigram_probability
from bigram import train_bigram_model, bigram_probability 
import pandas as pd
import re

def detect_outliers_with_bigram_model(paragraph, bigram_model, threshold):
    words = re.findall(r'\w+', paragraph.lower())
    outliers = []
    probability = []

    for i in range(len(words) - 1):
        bigram_prob = bigram_probability(bigram_model, words[i], words[i+1])
        probability.append(bigram_prob)
        if bigram_prob < threshold:
            outliers.append(1)
        else:
            outliers.append(0)
    return outliers, words, probability

def detect_outliers_with_trigram_model(paragraph, trigram_model, threshold):
    words = re.findall(r'\w+', paragraph.lower())
    outliers = []
    probability = []

    for i in range(len(words) - 2):
        trigram_prob = trigram_probability(trigram_model, words[i], words[i+1], words[i+2])
        probability.append(trigram_prob)
        if trigram_prob < threshold:
            outliers.append(1)
        else : outliers.append(0)
    return outliers, words, probability

def outlier_ngram(text_column,df):
    res = {
        'non_anomaly' : [],
        'anomaly' : [],
        'non_anomaly_probability': [],
        '%' : None
    }

    paragraphs = list(df[text_column])
    trigram_model = train_trigram_model(paragraphs)
    bigram_model = train_bigram_model(paragraphs)
    total = 0
    out = 0
    for paragraph in paragraphs :
        # labeling outliers based on bigram AND trigram
        threshold = 0.01  # adjust threshold as needed
        b_outliers, b_words, b_prob = detect_outliers_with_bigram_model(paragraph, bigram_model, threshold)
        t_outliers, t_words, t_prob = detect_outliers_with_trigram_model(paragraph, trigram_model, threshold)
        outliers = [b+t for b,t in zip(b_outliers,t_outliers)]
        outliers = [1 if item==2 else 0 for item in outliers]
        probabilities = [(b+t)/2 for b,t in zip(b_prob,t_prob)]
        
        # calculating % of outliers
        out = sum(outliers)
        total += len(b_words)
        
        # creating anomaly less data 
        non_anomaly = []
        anomaly = []
        non_anomaly_prob = []
        for i in range(len(outliers)):
            if not outliers[i] : 
                non_anomaly.append(b_words[i])
                non_anomaly_prob.append(probabilities[i])
            else : anomaly.append(b_words[i])

        res['non_anomaly'].append(' '.join(non_anomaly))
        res['anomaly'].append(' '.join(anomaly))
        res['non_anomaly_probability'].append(non_anomaly_prob)
    res['%'] = (out*100)/total
    
    return res