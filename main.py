import pandas as pd
from functools import partial

from wordcloud import WordCloud
import matplotlib.pyplot as plt

config = {
    'filename' : 'df_file',
    'text_cols' : ['Text']
}

# Define preprocess_row function with additional parameters
from preprocess import to_lower, lemmatize, remove_stopwords_frequency
from preprocess import remove_numbers, remove_single_char_words
def preprocess_row(row, stop_words):
    for col in config['text_cols']:
        row[col] = remove_single_char_words(row[col])
        row[col] = remove_numbers(row[col])
        row[col] = to_lower(row[col])
        row[col] = remove_stopwords_frequency(row[col], stop_words)
        row[col] = lemmatize(row[col])
        # row[col] = row[col].split(' ')
    return row

if __name__ =='__main__' :
    # READING DATA
    data_path = './data/'
    df = pd.read_csv(data_path+config['filename']+'.csv')

    # preprocessing 
    # preprocessing 
    # preprocessing 

    # # Calculate the top 10 stop words
    # from collections import Counter
    # d = Counter()
    # for col in config['text_cols']:
    #     for row in df[col]:
    #         d.update(row.split(' '))
    # stop_words = [word for word,freq in d.most_common()[:20]]

    # # Bind additional parameters to the preprocess_row function
    # preprocess_row_with_params = partial(preprocess_row, stop_words=stop_words)

    # # Apply preprocessing using threading
    # import concurrent.futures
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     results = list(executor.map(preprocess_row_with_params, df.to_dict(orient='records')))
    # df = pd.DataFrame(results)

    # df.to_csv('./data/'+config['filename']+'_preprocessed.csv',index=False)

    # preprocessing 
    # preprocessing 
    # preprocessing 

    # removing outliers 
    # removing outliers 
    # removing outliers 

    from outlier import outlier_ngram
    df = pd.read_csv('./data/'+config['filename']+'_preprocessed.csv')
    clean = {
        'Text' : [],
        'Label' : []
    }

    for label in df['Label'].unique()[:2]:
        res = outlier_ngram('Text',df[df['Label']==label])
        clean['Text']+=res['non_anomaly']
        clean['Label']+=[label]*len(res['non_anomaly'])
        
        # for wordcloud

        res = outlier_ngram('Text',df[df['Label']==label].sample(10))
        anomaly = ' '.join(res['anomaly']) 
        n = len(anomaly.split(' '))
        non_anomaly = ' '.join(res['non_anomaly'])
        m = len(non_anomaly.split(' '))
        non_anomaly = non_anomaly.split(' ')[:min(m,n)]
        non_anomaly = ' '.join(non_anomaly)
        
        text = anomaly+non_anomaly

        word_colors = {}
        word_sizes = {}

        for word in anomaly.split(' '):
            word_colors[word] = 'red'
            word_sizes[word] = 50

        probs = []
        for p in res['non_anomaly_probability']:
            probs+=p
        probs = probs[:min(m,n)]

        print (len(probs),len(non_anomaly))

        for word,prob in zip(non_anomaly.split(' '),probs):
            word_colors[word] = 'blue'
            word_sizes[word] = prob*100

        wordcloud = WordCloud(width=800, height=400, background_color='white')

        # Apply custom sizes and colors
        wordcloud.generate_from_frequencies({word: word_sizes[word] for word in word_sizes})
        wordcloud.recolor(color_func=lambda word, font_size, position, orientation, random_state=None, **kwargs: word_colors[word])

        # Display the generated word cloud
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title('Label:%s'%(str(label)))
        plt.axis('off')
        plt.show()

    df = pd.DataFrame(clean)
    # df.to_csv('./data/'+config['filename']+'_preprocessed.csv',index=False)
    
    # removing outliers 
    # removing outliers 
    # removing outliers 