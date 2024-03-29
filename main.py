import pandas as pd
from functools import partial

config = {
    'filename' : 'SMS_test',
    'text_cols' : ['Text']
}

# Define preprocess_row function with additional parameters
from preprocess import to_lower, lemmatize, remove_stopwords_frequency
def preprocess_row(row, stop_words):
    for col in config['text_cols']:
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

    # Calculate the top 10 stop words
    from collections import Counter
    d = Counter()
    for col in config['text_cols']:
        for row in df[col]:
            d.update(row.split(' '))
    stop_words = [word for word,freq in d.most_common()[:10]]

    # Bind additional parameters to the preprocess_row function
    preprocess_row_with_params = partial(preprocess_row, stop_words=stop_words)

    # Apply preprocessing using threading
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(preprocess_row_with_params, df.to_dict(orient='records')))
    df = pd.DataFrame(results)

    df.to_csv('./data/'+config['filename']+'_preprocessed.csv',index=False)

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
    for label in df['Label'].unique():
        res = outlier_ngram('Text',df[df['Label']==label])
        clean['Text']+=res
        clean['Label']+=[label]*len(res)
    df = pd.DataFrame(clean)

    df.to_csv('./data/'+config['filename']+'_preprocessed.csv',index=False)

    # removing outliers 
    # removing outliers 
    # removing outliers 