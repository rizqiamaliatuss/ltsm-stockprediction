import quandl
import pandas as pd 

auth_tok = "5DFie9M1JLAUNKr1dyyi"

tsla = quandl.get("WIKI/TSLA", trim_start = "2010-01-01", trim_end = "2020-10-01", authtoken=auth_tok)

#observation
#first row 
tsla.head()
#last row
tsla.tail()
#describe
tsla.describe()


tsla.to_csv('tsla2.csv')
tsla = pd.read_csv('tsla2', header = 0, index_col= 'Date', parse_date=True )

tsla.index
tsla.coloumns

ts = tsla['Close'][-10:]

type(ts)

print(tsla.loc[pd.Timestamp('2010-05-01'):pd.Timestamp('2010-10-01')].head())

print(tsla.loc['2011'].head())

print(tsla.iloc[22:43])

print(tsla.iloc[[22,43], [0, 3]])
