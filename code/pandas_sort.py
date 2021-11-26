import pandas as pd
import sys

df = pd.DataFrame({'a': [1, 2, 1], 'b': [1, 2, 3]})

df['count'] = df.groupby('a')['a'].transform('count')
print('df sorted: \n{}'.format(df))
df.sort_values('count', inplace=True, ascending=False)
print('df sorted: \n{}'.format(df))
unique_as = df.drop_duplicates(subset = ['a'])['a'].tolist()
print(unique_as)
print('df sorted: \n{}'.format(df))

