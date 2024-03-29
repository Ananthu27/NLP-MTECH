import pandas as pd
from functools import partial

config = {
    'filename' : 'df_file.csv',
    'text_cols' : ['Text']
}

# Define preprocess_row function with additional parameters
from preprocess import to_lower, lemmatize, remove_stopwords_frequency
def preprocess_row(row, stop_words):
    for col in config['text_cols']:
        row[col] = to_lower(row[col])
        row[col] = remove_stopwords_frequency(row[col], stop_words)
        row[col] = lemmatize(row[col])
    return row

if __name__ =='__main__' :
    # READING DATA
    data_path = './data/'
    df = pd.read_csv(data_path+config['filename'])

    # Calculate the top 20 stop words
    from collections import Counter
    d = Counter()
    for col in config['text_cols']:
        for row in df[col]:
            d.update(row.split(' '))
    stop_words = [word for word,freq in d.most_common()[:20]]

    # Bind additional parameters to the preprocess_row function
    preprocess_row_with_params = partial(preprocess_row, stop_words=stop_words)

    # Apply preprocessing using threading
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(preprocess_row_with_params, df.to_dict(orient='records')))
    df = pd.DataFrame(results)
    
    df.to_csv('./data/df_file_preprocessed.csv')