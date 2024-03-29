from trigram import train_trigram_model, detect_outliers_with_trigram_model
from bigram import train_bigram_model, detect_outliers_with_bigram_model 
import pandas as pd

def outlier_ngram(text_column,df):
    res = []

    paragraphs = list(df[text_column])
    trigram_model = train_trigram_model(paragraphs)
    bigram_model = train_bigram_model(paragraphs)
    total = 0
    out = 0
    for paragraph in paragraphs :
        # labeling outliers based on bigram AND trigram
        threshold = 0.0001  # adjust threshold as needed
        b_outliers = detect_outliers_with_bigram_model(paragraph, bigram_model, threshold)
        t_outliers = detect_outliers_with_trigram_model(paragraph, trigram_model, threshold)
        outliners = [b+t for b,t in zip(b_outliers,t_outliers)]
        outliners = [1 if item==2 else 0 for item in outliners]
        
        # calculating % of outliers
        out = sum(outliners)
        total += len(paragraph.split(' '))
        
        # creating anomaly less data 
        words = paragraph.split(" ")
        no_anomaly = []
        for i in range(len(outliners)):
            if not outliners[i] : no_anomaly.append(words[i])
        res.append(' '.join(no_anomaly))
    
    return res