import pandas
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
import os
import matplotlib.pyplot as plt


files = ['^GDAXI.csv', '^HSI.csv', '101670.KQ.csv', '^BSESN.csv']
total = 0
for file in files:

    data = pandas.read_csv(os.path.join('H:\Study\SEM - V\MP-II',file))
    #print(data.head())

    ret = data[['Close']].pct_change()
    # print(ret.head())
    ret['inter'] = 1
    # print(ret.head())
    data['inter'] = 1
    X = np.array(data[['Open','High','Low']][1:-1])
    # print(X[:6])

    y = (ret['Close'] > 0)[1:]
    # print(y[:5])
    N = len(y)

    n = 3964

    use = 60
    pre = 20
    max = 0
    for k in np.arange(0.38, 0.62, 0.01):
        sumT=0
        sum=0
        for i in range(0,N - use - pre,use):
            log = LogisticRegression(solver='lbfgs')
            log.fit(X[i:i+use], y[i:i+use])
            yp = log.predict_proba(X[i+use:i+use+pre])
            # sprint(yp)
            for j in range(len(yp)):
                if yp[j][1] < k:
                    yp[j][1] = False
                else:
                    yp[j][1] = True
                #print(f'yp = {yp[j][1]}, y = {y[j+i+use]}')
                if yp[j][1] == y[j+i+use]:
                    sumT += 1
                sum +=1
                new_max = sumT / sum
        if new_max > max:
            max = new_max
    total += max

    print(f'{file} = {max}')

print(f'avg:{total*100 / 4}')
