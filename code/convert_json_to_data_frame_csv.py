import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import pandas as pd

json_filename = '../data/ATC.json'
csv_filename = '../data/ATC.csv'

df = pd.read_json(json_filename, orient='records')
df.to_csv(csv_filename)

