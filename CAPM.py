#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 10:20:31 2021

@author: chewy2.0
S Craig
"""

# load modules
import numpy as np
import pandas as pd

# risk-free Treasury rate
R_f = 0.0175 / 252


# read in the market data
data = pd.read_csv('capm_market_data.csv')

#I like the format of the return / display better than the print statement
display(data.head(),data.tail())



#just checking the column names again
#data.columns

#data now equals the old data but just selected columns
#data = data[['spy_adj_close', 'aapl_adj_close']]

#Copy so that I can delete without lossing data forever
df=data.copy()

del df['date']

print(df)


#playing around to see if i can do it without internet help 
#I did it, but there is a function for that 
'''
df['daily_returns_spy']=np.nan
for i in range(len(df.spy_adj_close)-1):
    x=df.spy_adj_close[i]
    y=df.spy_adj_close[i+1]
    z = ((y-x)/x)
    df.daily_returns_spy[i+1]=z
    

#data['daily_returns_spy']=[((data.spy_adj_close[i+1]-data.spy_adj_close[i])/data.spy_adj_close[i])*100 for i in range(len(data.spy_adj_close)-1) if i>=1]

df['daily_returns_aapl']=np.nan
for i in range(len(df.aapl_adj_close)-1):
    a=df.aapl_adj_close[i]
    b=df.aapl_adj_close[i+1]
    c = ((b-a)/a) 
    df.daily_returns_aapl[i+1]=c  
#drop first row with nan
df=df.dropna()
'''

#pct_change() does it for you- then drop the nan in 1st row
returns=df.pct_change(axis=0)

returns.dropna(inplace=True)

#df[['daily_returns_spy','daily_returns_aapl']].head()
print(returns.head())


'''#save columns as individual arrays
spy_returns=df.daily_returns_spy.values
aapl_returns=df.daily_returns_aapl.values

#Print them with label so you know what they are
print('Spy Returns', spy_returns[:5])
print('\nAapl Returns', aapl_returns[:5])'''

#save columns as arrays
spy=returns.spy_adj_close.values
aapl=returns.aapl_adj_close.values

print('spy', spy[:5])
print('\naapl', aapl[:5])

#Can check to see if it is an array
#['column name'].values results in an array
type(spy)

'''#new arrays using line comprhension
spy_excess_returns=[s-R_f  for s in spy]

aapl_excess_returns=[a-R_f for a in aapl]'''

#subtract R_f from the spy and aapl
spy_xs = spy - R_f
aapl_xs=aapl - R_f

#take a look
display(spy_xs[:5],aapl_xs[:5])


'''#print the last 5 returns with labels so I know what they are
print('spy_excess',spy_excess_returns[-5:])
print('\naapl_excess', aapl_excess_returns[-5:])'''

print('spy excess', spy_xs[-5:])
print('\naapl excss', aapl[-5:])

#import so you can use it
import matplotlib.pyplot as plt

#scatter plot x, y , color of dots, edge color is the outline
plt.scatter(spy_xs, aapl_xs, color='orange', edgecolor='blue')

#show a grid
plt.grid()

#label the axises and give title
plt.xlabel('Spy Excess')
plt.ylabel('Aapl Excess')
plt.title('Aapl against Spy Daily Excess')
#returns plot? avoids other returns? 
plt.show()

#beta_hat = np.matmul(np.matmul(np.linalg.inv(np.matmul(x.transpose(), x)), x.transpose()), y)[0][0]

#reshape arrays to column vectors <- forces them to a singel column
x=spy_xs.reshape(-1,1)
y=aapl_xs.reshape(-1,1)

#multiply x'x - matmul is the matrix multiplier function
xtransx = np.matmul(x.transpose(),x)

#inverse is in the lin alg in np
xtransxinv=np.linalg.inv(xtransx)

#mutilpy the inverse by x'
xtransxinvxtrans= np.matmul(xtransxinv, x.transpose())

#to find beta then muliply by y (comes out as 1x1 array)
beta= np.matmul(xtransxinvxtrans,y)

#extract as a single value
#beta_hat= np.matmul(xtransxinvxtrans,y)[0][0]
beta_hat = beta[0][0]
print(beta_hat)


def beta_sensitivity(x,y):
    '''
    Purpose: Given to np arrays (x & y) as input, output list of tuples containing 
             observation row dropped and beta estimate
    
    INPUT:
    x      np array
    y      np array
    
    OUTPUT:
    out    list of tuples
    '''
    
    out = []
    #x.shape[0] gives the first term in the size of x
    #shape usually returns rows,columns so this will give # of rows in x
    sz = x.shape[0]
    
    
    for ix in range(sz):
        #this will delete observation i from array x, and make it a column vector
        xx = np.delete(x,ix).reshape(-1,1)
        yy=np.delete(y,ix).reshape(-1,1)
        #Calculate B hat using oneliner
        bi=np.matmul(np.matmul(np.linalg.inv(np.matmul(xx.transpose(), xx)), xx.transpose()), yy)[0][0]
        out.append((ix,bi))
    
    return out

betas = beta_sensitivity(x, y)
betas[:5]

