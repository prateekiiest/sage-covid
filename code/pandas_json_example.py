import pandas as pd

df = pd.DataFrame([['a', 'b'], ['c', 'd']],
                  index=['row 1', 'row 2'],
                  columns=['col 1', 'col 2'])
print(df)

print(df.to_json(orient='split'))

json_file = open('../data/ATC.json', 'r')

df = pd.read_json('../data/ATC.json', orient='records')
print(df)
