import pandas
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
import os
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

files = ['^GDAXI.csv', '^HSI.csv', '101670.KQ.csv']
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


    old_y = ret['Close'][1:]
    N = len(old_y)
    lab_enc = preprocessing.LabelEncoder()
    y = lab_enc.fit_transform(old_y)
    n = 2
    use = 40
    pre = 40
    max = 0
    step = 0.25
    errors = []
    y_final=[]
    for k in zip(range(100,300,100), range(2,10,4)):
        sumT=0
        sum=0
        yp = []
        for i in range(0,N - use - pre,use):
            rf = RandomForestClassifier(n_estimators=k[0])
            rf.fit(X[i:i+use], y[i:i+use])
            yp = rf.predict(X[i+use:i+use+pre])
            acc = accuracy_score(y[i+use:i+use+pre], yp)
            errors.append(acc)
            if k[0] == 100:
                y_final.extend(yp)
        # print(np.sum(errors) / len(errors))
    acc = np.sum(errors) / len(errors)
    print(f'Max for file {file} is {1- acc*100}.')
    total += 1 - acc*100

print(f'Avg = {total*100/4}')
