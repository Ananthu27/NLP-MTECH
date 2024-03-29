import pandas as pd

config = {
    'filename' : 'df_file.csv',
    'text_cols' : ['Text']
}

# READING DATA
data_path = './data/'
df = pd.read_csv(data_path+config['filename'])
df = df[:5]

# 1 LOWERCASE
from preprocess import to_lower
for col in config['text_cols'] : 
    df[col] = df[col].apply(to_lower)

# 3 LEMMATIZATION 
from preprocess import lemmatize
for col in config['text_cols']:
    df[col] = df[col].apply(lemmatize)

print (df.head())