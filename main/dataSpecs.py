import pandas as pd
fname = '../datasets/binary_dataset_Copy.csv'
data = pd.read_csv(fname, encoding = "ISO-8859-1")
print(data["tweet_class"].value_counts())