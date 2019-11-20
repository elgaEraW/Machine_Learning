import pandas
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import math

files = ['^GDAXI.csv', '^HSI.csv', '101670.KQ.csv', '^BSESN.csv']
total = 0
n = 4
for file in files:

    data = pandas.read_csv(os.path.join('H:\Study\SEM - V\MP-II',file), header=None)
    #print(data.head())

    # ret = data[['Close']].pct_change()
    # #print(ret.head())
    # ret['inter'] = 1
    #print(ret.head())
    X = []
    for i in range(5663):
        X.append([])
    for i in range(3):
        temp = data[i+1]
        # print(len(temp))
        # print(temp)
        for j in range(1,len(temp)-1):
            X[j-1].append(temp[j-1])
    # X = data[1:4][1:]
    # X = X.to_csv(header=None, index=False)
    # y = ret['Close'][1:]
    y = data[4][1:]
    N = len(y)
    for i in range(1,N-1):
        y[i] = (float(y[i+1]) - float(y[i]))/float(y[i])

    y[N-1] = 1
    y[N] = 1
    # print(y)

    X = X[1:]
    y_final = []
    # print(y)
    #print(y[:5])
    # svr = SVR(kernel='rbf',gamma=0.25, C=2)
    # svr.fit(X[:n],y[:n])

    use = 40
    pre = 40
    max = 0
    step = 0.25
    errors = []
    for k in zip(np.arange(0.25, 4, step), range(2,10,2)):
        sum=0
        yp = []
        for i in range(0,N - use - pre,use):
            error=0
            svr = SVR(kernel='rbf', gamma=k[0], C=k[1])
            svr.fit(X[i:i+use], y[i:i+use])
            yp = svr.predict(X[i+use:i+use+pre])
            # print(yp)
            sum = 0
            for tup in zip(yp,y[i+use:i+use+pre]):
                sum += (abs(tup[0] ** 2 - tup[1] ** 2))
            # print(sum)
            error += sum * 10
            # print(f'length of yp = {len(yp)}')
            if k[0] == 0.25:
                y_final.extend(yp)
            # print(f'length of y_final = {len(y_final)}')
            # error = np.sqrt(mean_squared_error(y[i+use:i+use+pre], yp))
            # /errors.append(1-error)
        step = k[0]
        # print(np.sum(errors) / len(errors))
    # print(f'Max for file {file} is {error} %.')
    total +=100 - error*100

    temp = [i for i in range(N)]


    plt.figure()
    plt.plot(temp, y)
    N_final = len(y_final)
    print(f'final = {N_final - N}')
    y_final.extend(y[N_final:])
    print(f'len of y_final = {len(y_final)}, len of temp = {len(temp)}')
    plt.figure()
    y_final = [np.max(y)*i for i in y_final]
    plt.plot(temp, y_final)
    # plt.show()

print(f'Avg = {total/4} %')
