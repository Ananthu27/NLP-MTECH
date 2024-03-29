import pandas as pd

config = {
    'filename' : 'df_file.csv',
    'text_cols' : ['Text']
}

# READING DATA
data_path = './data/'
df = pd.read_csv(data_path+config['filename'])

# 1 LOWERCASE
for col in config['text_cols'] : 
    df[col] = df[col].str.lower()

print (df.head())