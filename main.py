import pandas as pd

data_path = './data/'

df = pd.read_csv(data_path+'df_file.csv')
# df = pd.read_excel(data_path+'SMS_train.xls')

print (df.head())
