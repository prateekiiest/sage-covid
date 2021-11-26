import pandas as pd
import sys
import numpy as np
from datetime import datetime

print(datetime.now())

df=pd.DataFrame({ 'id':[1,1,1,1,2,2,2,2],
                   'a':range(8), 'b':range(8,0,-1) })

print(df)
#print(np.isnan(df.query('id==3')['a'].min()))

#print(df['a'].min())

#for x in df['a']:
#	print(x)
for i in df.drop_duplicates(subset = ['id'])['id']:

	print(i)


#.loc[:, 'id'])
sys.exit(0)

for i in df.groupby('id').loc[:, 'id']:
	print(i)

sys.exit(0)

df_mean = df.groupby('id')['a'].mean()
#df_max['type'] = 'max'
print(df_mean)
for i in df_mean.keys():
	print(i)


