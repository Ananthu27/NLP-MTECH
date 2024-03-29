import pandas as pd

config = {
    'filename' : 'df_file.csv',
    'text_cols' : ['Text']
}

if __name__ =='__main__' :
    
    # READING DATA
    data_path = './data/'
    df = pd.read_csv(data_path+config['filename'])
    df = df[:100]

    # 1 LOWERCASE
    from preprocess import to_lower
    for col in config['text_cols'] : 
        df[col] = df[col].apply(to_lower)

    # 3 LEMMATIZATION 
    from preprocess import lemmatize

    def lemmatize_row(row):
        for col in config['text_cols'] :
            row[col] = lemmatize(row[col])
        return row

    # applying lemmatization in a thread pool
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(lemmatize_row, df.to_dict(orient='records')))
    df = pd.DataFrame(results)
    print (df.head())